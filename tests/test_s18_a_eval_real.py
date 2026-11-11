"""S-A-EVAL на реальных выходах (гейты E.G1–E.G3, только чтение; polnota.md п. 10): таблицы `S_A_*.csv`, `S_A_REPORT.md` и ключ `S.A.eval` проверяются независимым кодом теста — Σ вкладов = Δ (1e-12),
лаг NW = 8, B = 10 000, Холм N = 831 / 3367 и m = 12 пересчитан из p DM, Δ = 100 (Q_h / Q_b − 1) по QLIKE строки, пороги 0,415 / 0,50 / 1,00 в строках GUARD и выполнение пересчитано, решение выведено из GUARD с порядком A2 → A1,
«подтверждено» только по своей строке (Δ < 0, p Холма N < 0,05, p SPA < 0,05), пометка и даты / n в отчёте; `E.check()` и `D.check()` ничего не меняют (содержимое до и после). Тест не пишет и не меняет реальные файлы."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from models_v2.s18 import a_doc as D
from models_v2.s18 import a_eval as E
from models_v2.s18 import a_eval_yr as Y

NEED = [E.REP / n for n in E.OUTS] + [E.REP / "S_A_REPORT.md", E.CKPT]
pytestmark = pytest.mark.skipif(not all(p.exists() for p in NEED), reason="выходы S-A-EVAL ещё не записаны (`a_eval --write`, `a_doc --write`)")


def snap() -> dict:
    ps = [E.CKPT, E.R.TRIALS] + [E.REP / n for n in E.OUTS] + [E.REP / "S_A_REPORT.md"] + sorted(E.A_DIR.glob("*.parquet"))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in ps if p.exists() and p.is_file()}


@pytest.fixture(autouse=True)
def real_files_untouched():
    before = {str(p): p.read_bytes() for p in NEED + sorted(E.A_DIR.glob("*.parquet")) + [E.R.TRIALS] if p.exists()}
    yield
    assert {str(p): p.read_bytes() for p in NEED + sorted(E.A_DIR.glob("*.parquet")) + [E.R.TRIALS] if p.exists()} == before


def rd(k: str) -> pd.DataFrame:
    return pd.read_csv(E.REP / ("S_A_%s.csv" % k.upper()), comment="#")


def holm(p, N: int) -> np.ndarray:
    p = np.asarray(p, float)
    out, run = np.full(len(p), np.nan), 0.0
    f = np.flatnonzero(np.isfinite(p))
    for r, i in enumerate(f[np.argsort(p[f], kind="stable")]):
        run = max(run, min(1.0, (N - r) * p[i]))
        out[i] = run
    return out


def test_stats_table_real_known_formulas() -> None:
    s = rd("stats")
    m = s[s["kind"] == "main"].reset_index(drop=True)
    assert len(m) == 12 and (m["window"] == "M").all() and (m["lag"] == 8).all() and (m["n"] == 2198).all() and (m["first"] == "2018-01-04").all() and (m["last"] == "2026-11-10").all()      # лаг NW-94 = 8
    assert (s["B"] == 10000).all() and (s["n_holm"] == 831).all() and (s["n_holm_wide"] == 3367).all() and (s["m_family"] == 12).all()
    assert m["delta_pct"].to_numpy() == pytest.approx(100 * (m["qlike_h"] / m["qlike_b"] - 1).to_numpy(), rel=1e-9, abs=1e-8)         # знак и база: Δ от QLIKE базы (округление CSV 12 знаков)
    for col, N in (("p_holm_m12", 12), ("p_holm_N", 831), ("p_holm_Nwide", 3367)):
        assert m[col].to_numpy() == pytest.approx(holm(m["p_dm"].to_numpy(), N), rel=1e-9, nan_ok=True), col
    for r in m.itertuples():                                                             # № 158: «подтверждено» — только по своей строке
        conf = r.delta_pct < 0 and r.p_holm_N < 0.05 and r.p_spa_c < 0.05
        assert ("подтверждено (" in r.verdict and "не подтверждено" not in r.verdict) == conf, r.hypothesis
        assert ("численно лучше базы" in r.verdict) == (r.delta_pct < 0), r.hypothesis          # № 125/140
    assert m["hypothesis"].tolist() == ["%s|%s" % (c, b) for c in ("S-A0", "S-A1", "S-A2") for b in ("X1", "W5-G", "K2", "K3")]
    assert (m.loc[m["base"] == "X1", "series"].str.startswith("X1_A(")).all() and not m.loc[m["base"] != "X1", "series"].str.startswith("X1_A(").any()      # слои согласованы


def test_years_identity_and_guard_thresholds_real() -> None:
    y = rd("years")
    p = y[y["base"] != "-"]
    gap = max(abs(g["contrib_pct"].sum() - g["delta_total_pct"].iloc[0]) for _, g in p.groupby(["series", "base", "kind"]))
    assert gap < 1e-9                                                                    # CSV хранит 12 значащих цифр (float_format %.12g): на файле — 1e-9; 1e-12 в памяти проверяет E.check (identity)
    g = rd("guard")
    thr = {"У1-S": 0.415, "У2-S": 0.5, "У3-S": 1.0}
    c = g[~g["id"].isin(["кандидат", "решение"])]
    assert len(c) == 20 and set(c["candidate"]) == {"S-A2", "S-A1"}
    for r in c.itertuples():
        if r.id in thr:
            assert float(r.threshold) == thr[r.id] and bool(r.passed) == (float(r.value) <= thr[r.id]), (r.candidate, r.pair, r.id)   # константы спецификации, не из результата
        if r.id in ("М1", "М2"):
            assert bool(r.passed) == (float(r.value) < 0)
    dec = g[g["id"] == "решение"]["value"].iloc[0]
    assert Y.decide(g) == dec and "X1 и W5-G остаются контрольными версиями" in dec
    ok = {k: bool(g[(g["candidate"] == k) & ~g["id"].isin(["кандидат", "решение"])]["passed"].all()) for k in ("S-A2", "S-A1")}
    want = "S-A2" if ok["S-A2"] else "S-A1" if ok["S-A1"] else None                       # порядок A2 → A1
    assert dec.startswith("новая контрольная версия: " + want) if want else dec.startswith("не прошёл")


def test_table_mcs_checkpoint_and_report_real() -> None:
    t = rd("table")
    assert (t["n"] == 2198).all() and (t["first_tplus"] == "2018-01-04").all() and (t["last_tplus"] == "2026-11-10").all()
    for b, col in (("X1", "d_X1_pct"), ("W5-G", "d_W5-G_pct"), ("K2", "d_K2_pct"), ("K3", "d_K3_pct")):
        x = t.set_index("series")
        assert x[col].to_numpy() == pytest.approx(100 * (x["qlike"] / x.at[b, "qlike"] - 1).to_numpy(), rel=1e-9, abs=1e-8)      # Δ к базе b
    mc = rd("mcs")
    assert (mc["n_used"] == 2198).all() and set(mc["series"]) == {"S-A0", "S-A1", "S-A2", "W5-G", "X1", "W8-G", "K2", "K3", "B_mix"}
    ck = json.loads(E.CKPT.read_text(encoding="utf-8"))["S.A.eval"]
    assert ck["n_holm"] == 831 and ck["n_holm_wide"] == 3367 and ck["m_family"] == 12 and ck["boot"] == 10000 and ck["lag_main"] == 8 and ck["n"] == 2198
    assert ck["outputs"] == {n: hashlib.sha256((E.REP / n).read_bytes()).hexdigest() for n in E.OUTS}      # ключ = хэшам таблиц
    text = (E.REP / "S_A_REPORT.md").read_text(encoding="utf-8")
    assert text.splitlines()[0] == D.MARK and "исследовательский" in text.splitlines()[0] and "не независимый OOS" in text.splitlines()[0]
    assert text.split("\n## ")[1].startswith("1. Таблица") and "D_M: t⁺ 2018-01-04…2026-11-10, n = 2198" in text.split("\n## ")[1]


def test_e_g1_check_and_doc_check_read_only_and_pass() -> None:
    h0 = snap()
    assert E.check() == [] and snap() == h0                                              # E.G1: проблем нет, выходы до и после равны
    assert D.check() == [] and snap() == h0                                              # E.G3


def test_e_g2_check_spa_known_answer_real_sigma() -> None:
    assert E.check_spa_cli() == []
