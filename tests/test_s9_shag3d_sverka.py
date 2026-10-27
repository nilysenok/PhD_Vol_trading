"""Тесты scripts/analysis/s9_shag3d_sverka.py (pick_points, check_vs_shag0) и обязательной проверки сдвига в запуске D.
Мутации в памяти: Шарп A или к3 искажён на 1e-3; неверный выбор 5 точек; удалён вызов проверки сдвига — тесты падают."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import s9_shag3d_sverka as V  # noqa: E402
import s9_shag_common as C  # noqa: E402
import s9_shag_db as DB  # noqa: E402
from fastbt import shag0  # noqa: E402

N = 300
IDX = pd.date_range("2018-01-01", periods=N, freq="B")
RET = np.random.RandomState(7).normal(0.0004, 0.01, N)
K3 = 0.5 * np.ones(N)


def frame(ret):
    return pd.DataFrame(dict(excess=ret, total=ret, exposure=np.ones(N), turnover=np.full(N, 0.01)), index=IDX)


def fake_eng():
    return SimpleNamespace(A_row=np.ones(N), r={"pair": None}, pairs=None,
                           port=lambda pos, comm=0.0: frame(RET * np.asarray(pos)))


def sharpe_of(pos):
    return round(shag0.stats(frame(RET * pos), np.ones(N, bool))["sharpe"], 4)


def tables(tmp_path, monkeypatch, d_a=0.0, d_k3=0.0):
    monkeypatch.setattr(C, "OUT", tmp_path)
    monkeypatch.setattr(V.fk, "k3_positions", lambda A, pair, pairs: K3)
    a = pd.DataFrame([dict(row="A", comm_pct=c, sharpe=sharpe_of(np.ones(N)) + d_a) for c in (0.04, 0.035)])
    k = pd.DataFrame([dict(row="к3: A без S5/S6", comm_pct=c, sharpe=sharpe_of(K3) + d_k3) for c in (0.04, 0.035)])
    for df, name in ((a, "shag0_A_metriki.csv"), (k, "shag0_k3.csv")):
        df.to_csv(tmp_path / name, sep=";", decimal=",", index=False, encoding="utf-8-sig")


def test_check_vs_shag0_ok_on_consistent_tables(tmp_path, monkeypatch):
    tables(tmp_path, monkeypatch)
    for comm in (0.0004, 0.00035):
        rows = V.check_vs_shag0(fake_eng(), comm)
        assert len(rows) == 2 and all(r["metriki_do_4_znaka"] for r in rows)


def test_check_vs_shag0_catches_a_distorted_by_1e3(tmp_path, monkeypatch):
    tables(tmp_path, monkeypatch, d_a=1e-3)
    with pytest.raises(AssertionError):
        V.check_vs_shag0(fake_eng(), 0.0004)


def test_check_vs_shag0_catches_k3_distorted_by_1e3(tmp_path, monkeypatch):
    tables(tmp_path, monkeypatch, d_k3=1e-3)
    with pytest.raises(AssertionError):
        V.check_vs_shag0(fake_eng(), 0.0004)


def _pts(extra_scale=True):
    g = np.ones(3)
    base = [("D1", "W126_p70", g, False, 0.3, True), ("D5", "x30", g, False, 0.3, True),
            ("D6lag1", "W252_p70", g, False, 0.3, True), ("D6lag1", "W252_p80", g, False, 0.3, True),
            ("D7", "W126_g1", g, True, 0.0, True), ("D1", "W126_p80", g, False, 0.3, True)]
    return base if extra_scale else base[:4]


def test_pick_points_exact_five(monkeypatch):
    monkeypatch.setattr(DB, "points", lambda *a: _pts())
    got = [(f, lab) for f, lab, _ in V.pick_points(None)]
    assert got == [("D6lag1", "W252_p70"), ("D5", "x30"), ("D6lag1", "W252_p80"), ("D1", "W126_p70"), ("D7", "W126_g1")]


def test_pick_points_wrong_selection_is_caught(monkeypatch):
    monkeypatch.setattr(DB, "points", lambda *a: _pts())
    monkeypatch.setattr(V, "PICK", [("D1", "W126_p70"), ("D5", "x30")])    # мутация: неверный выбор
    got = [(f, lab) for f, lab, _ in V.pick_points(None)]
    assert got != [("D6lag1", "W252_p70"), ("D5", "x30"), ("D6lag1", "W252_p80"), ("D1", "W126_p70"), ("D7", "W126_g1")]
    assert len(got) <= 5


def csv_sdvig(tmp_path, day_ok=True, win_ok=False):
    rows = [dict(ryad="hyb_h1", ok=day_ok, argmax_prirascheniya=0.0, cel="RV_{d+k} дневная"),
            dict(ryad="hyb_h5", ok=win_ok, argmax_prirascheniya=-1.0, cel="y_{d+k} окно")]
    p = tmp_path / "proverka_sdvig.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_sdvig_zhurnal_first_line_and_window_does_not_block(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "OUT", tmp_path)
    line = DB.sdvig_zhurnal("D", "hyb", csv_sdvig(tmp_path))
    first = (tmp_path / "shag3D_zhurnal_hyb.txt").read_text(encoding="utf-8").splitlines()[0]
    assert first == line.splitlines()[0] and "НЕ ПРОЙДЕНА" not in line


def test_sdvig_zhurnal_day_row_not_ok_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "OUT", tmp_path)
    with pytest.raises(SystemExit):
        DB.sdvig_zhurnal("D", "hyb", csv_sdvig(tmp_path, day_ok=False))


def test_run_calls_sdvig_check_before_counting(tmp_path, monkeypatch):
    """Удаление вызова sdvig_zhurnal из run() роняет тест: заглушка обязана сработать до любого счёта."""
    monkeypatch.setattr(C, "OUT", tmp_path)
    calls = []

    def stub(podhod, sg, path=None):
        calls.append((podhod, sg))
        raise RuntimeError("стоп после проверки сдвига")

    monkeypatch.setattr(DB, "sdvig_zhurnal", stub)
    monkeypatch.setattr(DB, "run_grid", lambda *a: pytest.fail("счёт начался до проверки сдвига"))
    with pytest.raises(RuntimeError, match="стоп после проверки"):
        DB.run("D", "hyb", 10, eng=SimpleNamespace())
    assert calls == [("D", "hyb")]
