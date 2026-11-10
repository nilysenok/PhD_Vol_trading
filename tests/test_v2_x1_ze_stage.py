"""Лист ZE (Ж-247 п. 4), стадия и отчёт: отчёт — генератор из таблиц (первая строка — статус; подменённое число в отчёте, `ZE_CRIT.csv`, parquet ловится сверкой `_check_files`); хэш чужих выходов не включает файлы и ключи ZE, но видит чужие;
`--check` только читает (мутация: запись файла внутри `--check` ловится хэшем до/после); отказ Y5, Y6 и счёта без предусловия."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import models_v2.s16  # noqa: F401
from models_v2.s16 import x1_ze_crit as CR, x1_ze_doc as DOC, x1_ze_inp as I, x1_ze_num as N, x1_ze_stage as S, x1_ze_synth as SZ, x1_ze_walk as W

NAMES = ("K1", "K2", "K3", "V4:NTP2", "V6:A2", "V7:cfg6", "V7:cfg8", "X1", "Y1", "Y2", "Y3", "ZA", "ZB", "ZB0")


@pytest.fixture(scope="module")
def synth():
    x1, D = SZ.make(2300, 11, p_e=0.3)
    rng = np.random.default_rng(3)
    X1, tp, d = x1["F"].to_numpy(float), pd.DatetimeIndex(x1["t_plus"]), pd.DatetimeIndex(x1["d"])
    zs = {k: W.walk(x1, D, k) for k in W.KINDS}
    s = {k: X1 * np.exp(rng.normal(0, 0.05, len(X1))) for k in NAMES} | {"X1": X1} | {k: f["F"].to_numpy(float) for k, f in zs.items()}
    t = CR.main_tables(x1["rv"].to_numpy(float), s, tp, d, D["e"], zs)
    pre = dict(n=179, corr_G_v=0.69, corr_G_s=-0.17, corr_G_x_info=0.15, max_abs=0.69, limit=0.8, ok=True, g_mean=0.1, g_std=0.3)
    num = dict(n_dates=len(d), first="2018-01-03", last="2026-12-01", pairs_at_activation=124, min_pairs=120, n_ext_at_activation=40, min_ext=30, n_inactive={"ZE": 124, "ZEp": 124}, n_extreme_days=int(D["e"].sum()), n_degenerate_days=0, lams=[0.0, 0.001, 0.01, 0.1],
               v_ext=60, n_holm=751, n_wide=3207, configs_spent=2, registered_at="ZE t0; ZEp t0", precondition=pre, precondition_at="t0", notes=["заметка"])
    return zs, t, num


def test_report_first_line_is_status_and_contains_all_sections(synth) -> None:
    _, t, num = synth
    text = DOC.build(t, num)
    assert text.splitlines()[0].startswith("разработка, историческая пошаговая проверка, не независимая; ZE — после просмотра Y1–Y4, Z, ZD")
    for h in ("K1–K6", "без 2022", "2023_2025", "Y5 — не запускалось", "Выученные параметры по годам", "p Холма N = 751", "ZEp"):
        assert h in text
    assert text == DOC.build(t, num)                                   # детерминированный генератор


def test_tamper_report_crit_csv_and_parquet_are_caught_and_honest_is_clean(synth, tmp_path, monkeypatch) -> None:
    zs, t, num = synth
    monkeypatch.setattr(S, "REP", tmp_path), monkeypatch.setattr(S, "OUT", tmp_path / "o"), monkeypatch.setattr(S, "REPORT", tmp_path / "ZE_REPORT.md")
    (tmp_path / "o").mkdir()
    for k, f in zs.items():
        f.to_parquet(tmp_path / "o" / (k.lower() + ".parquet"), index=False)
    for k, v in S.TABLES.items():
        (tmp_path / (k + ".csv")).write_text(I.csv_text(t[v]), encoding="utf-8")
    S.REPORT.write_text(DOC.build(t, num), encoding="utf-8")
    ck = dict(sha256={k: I.sha_file(tmp_path / "o" / (k.lower() + ".parquet")) for k in zs})
    ev, ru = dict(numbers=S.SA._json(S.eval_numbers(t)), run=S.SA._json(num)), dict(rules=S.SA._json(S.rules(t)), sha256={k: I.sha_file(tmp_path / (k + ".csv")) for k in S.TABLES})
    bad = []
    S._check_files(t, num, zs, ck, ev, ru, bad)
    assert bad == []
    crit = tmp_path / "ZE_CRIT.csv"
    crit.write_text(crit.read_text(encoding="utf-8").replace("K1,", "K1,9", 1), encoding="utf-8")        # подменённое число в ZE_CRIT.csv
    rep = tmp_path / "ZE_REPORT.md"
    rep.write_text(rep.read_text(encoding="utf-8").replace("751", "749", 1), encoding="utf-8")           # подменённое число в отчёте
    bad = []
    S._check_files(t, num, zs, ck, ev, ru, bad)
    assert any("ZE_CRIT.csv" in b for b in bad) and any("ZE_REPORT.md" in b for b in bad)
    zs2 = {k: f.assign(F=f["F"] * 1.001) for k, f in zs.items()}                                          # подменённые прогнозы
    bad = []
    S._check_files(t, num, zs2, ck, ev, ru, bad)
    assert any("ze.parquet" in b for b in bad)


def test_rep_digest_ignores_ze_files_and_keys_but_sees_foreign_changes_and_trials(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(N, "REP", tmp_path)
    (tmp_path / "ALL_TRIALS.csv").write_text("id\nY1\n", encoding="utf-8")
    (tmp_path / "Z_CRIT.csv").write_text("x\n", encoding="utf-8")
    cp = tmp_path / "CHECKPOINT.json"
    cp.write_text(json.dumps({"Z": {"a": 1}, "ZE": {"b": 1}}), encoding="utf-8")
    h0 = N.rep_digest()
    (tmp_path / "ZE_CRIT.csv").write_text("new\n", encoding="utf-8")
    cp.write_text(json.dumps({"Z": {"a": 1}, "ZE": {"b": 2}, "ZE.eval": {}}), encoding="utf-8")
    assert N.rep_digest() == h0
    (tmp_path / "Z_CRIT.csv").write_text("changed\n", encoding="utf-8")
    assert N.rep_digest() != h0
    (tmp_path / "Z_CRIT.csv").write_text("x\n", encoding="utf-8")
    cp.write_text(json.dumps({"Z": {"a": 2}, "ZE": {"b": 2}}), encoding="utf-8")
    assert N.rep_digest() != h0
    cp.write_text(json.dumps({"Z": {"a": 1}}), encoding="utf-8")
    (tmp_path / "ALL_TRIALS.csv").write_text("id\nY1\nZE\n", encoding="utf-8")
    assert N.rep_digest() != h0


@pytest.mark.parametrize("stage", ["Y5", "Y6"])
def test_main_refuses_y5_y6(stage) -> None:
    with pytest.raises(SystemExit):
        S.main(["--stage", stage])


def test_write_refuses_without_precondition(monkeypatch) -> None:
    monkeypatch.setattr(S, "read_cp", lambda: {})
    with pytest.raises(SystemExit):
        S.write()


def test_real_check_reads_only_and_mutation_write_in_check_is_caught(monkeypatch) -> None:
    cp = I.read_cp()
    if "ZE.rules" not in cp:
        pytest.skip("прогнозы ZE ещё не записаны")
    cache = S.compute()
    monkeypatch.setattr(S, "compute", lambda: cache)
    assert S.check() == []
    probe = I.REP / "P0_probe_mutant.txt"

    def writing_compute():                                            # мутация: --check пишет файл в каталог чужих отчётов
        probe.write_text("x", encoding="utf-8")
        return cache
    monkeypatch.setattr(S, "compute", writing_compute)
    try:
        assert any("изменил выходы" in b for b in S.check())
    finally:
        probe.unlink(missing_ok=True)
