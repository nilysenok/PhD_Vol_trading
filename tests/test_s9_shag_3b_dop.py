"""Тесты scripts/analysis/s9_shag_3b_dop.py на синтетике: сверка A/к3 с шагом 0, журнал сдвига, сверка слоёв.
Мутация в памяти (искажение Шарпа на 1e-3, ok=False, сдвиг excess) обязана ронять сверку."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import s9_shag_3b_dop as D  # noqa: E402
import s9_shag0  # noqa: E402
import s9_shag_common as C  # noqa: E402
from fastbt import shag0  # noqa: E402
from fastbt import shag_c_tab as T  # noqa: E402

N = 300
IDX = pd.date_range("2018-01-01", periods=N, freq="B")
RET = np.random.RandomState(7).normal(0.0004, 0.01, N)


def frame(ret):
    return pd.DataFrame(dict(excess=ret, total=ret, exposure=np.ones(N), turnover=np.full(N, 0.01)), index=IDX)


def fake_eng():
    return SimpleNamespace(A_row=np.ones(N), dates=IDX, port=lambda pos, comm=0.0: frame(RET * np.asarray(pos)))


def write_tables(tmp_path, monkeypatch, d_a=0.0, d_k3=0.0):
    monkeypatch.setattr(C, "OUT", tmp_path)
    sh = round(shag0.stats(frame(RET), np.ones(N, bool))["sharpe"], 4)
    k3 = -0.5905
    a = pd.DataFrame([dict(row="A", comm_pct=c, sharpe=sh + d_a) for c in (0.04, 0.035)])
    k = pd.DataFrame([dict(row="к3: A без S5/S6", comm_pct=c, sharpe=k3 + d_k3) for c in (0.04, 0.035)])
    for df, name in ((a, "shag0_A_metriki.csv"), (k, "shag0_k3.csv")):
        df.to_csv(tmp_path / name, sep=";", decimal=",", index=False, encoding="utf-8-sig")
    for sg in C.SIGMAS:
        rows = [dict(comm_pct=c, sharpe=sh - 0.5, d_vs_A=-0.5, sharpe_k3=k3) for c in (0.04, 0.035)]
        pd.DataFrame(rows).to_csv(tmp_path / f"shag3B_otbor_{sg}.csv", sep=";", decimal=",", index=False,
                                  encoding="utf-8-sig")


def test_a_k3_ok_on_consistent_tables(tmp_path, monkeypatch):
    write_tables(tmp_path, monkeypatch)
    rows = D.a_k3_vs_shag0(fake_eng())
    assert len(rows) == 6 and all(r["ok"] for r in rows)


def test_a_distorted_by_1e3_is_not_ok(tmp_path, monkeypatch):
    write_tables(tmp_path, monkeypatch, d_a=1e-3)
    assert not any(r["ok"] for r in D.a_k3_vs_shag0(fake_eng()))


def test_k3_distorted_by_1e3_is_not_ok(tmp_path, monkeypatch):
    write_tables(tmp_path, monkeypatch, d_k3=1e-3)
    assert not any(r["ok"] for r in D.a_k3_vs_shag0(fake_eng()))


def day(ok, k=1):
    return dict(ryad=f"hyb_h{k}", ok=ok, argmax_prirascheniya=0.0, cel="RV_{d+k} дневная")


def win(ok):
    return dict(ryad="hyb_h5", ok=ok, argmax_prirascheniya=0.0, cel="y_{d+k} окно")


def test_shift_day_row_not_ok_blocks():
    assert "НЕ ПРОЙДЕНА" in T.sdvig_line(pd.DataFrame([day(True), day(False, 5)]))
    assert "НЕ ПРОЙДЕНА" in T.sdvig_line(pd.DataFrame([day(False)]))


def test_shift_window_row_not_ok_does_not_block():
    line = T.sdvig_line(pd.DataFrame([day(True), win(False)]))
    assert "НЕ ПРОЙДЕНА" not in line and "все допущены" in line


def test_shift_bad_argmax_blocks_and_empty_blocks():
    bad = day(True)
    bad["argmax_prirascheniya"] = 1.0
    assert "НЕ ПРОЙДЕНА" in T.sdvig_line(pd.DataFrame([bad]))
    assert "НЕ ПРОЙДЕНА" in T.sdvig_line(pd.DataFrame(columns=["ryad", "ok", "cel"]))


def test_layer_check_passes_and_catches_divergence(monkeypatch):
    eng, pos = fake_eng(), np.ones(N)
    monkeypatch.setattr(s9_shag0, "slow_port", lambda e, p, comm: frame(RET * np.asarray(p)))
    ok = D.sverka(eng, [("точка", pos)], 0.0004)
    assert ok[0]["metriki_do_4_znaka"] is True or bool(ok[0]["metriki_do_4_znaka"])
    bent = RET.copy()
    bent[5:] += 1e-3   # медленный слой уехал: метрики и max|Δ| обязаны заметить
    monkeypatch.setattr(s9_shag0, "slow_port", lambda e, p, comm: frame(bent * np.asarray(p)))
    bad = D.sverka(eng, [("точка", pos)], 0.0004)
    assert not bad[0]["metriki_do_4_znaka"] and float(bad[0]["max_abs_excess"]) > 5e-4
