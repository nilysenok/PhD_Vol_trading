"""S-C-EVAL, `cm_eval_yr` + `cm_eval_chk` (S_C_SPEC разд. 4.3, 4.6; mutations.md № 125, 140, 158): решение NEWCTRL и STRUCT выводится кодом из строк GUARD; порядок кандидатов S-CK → S-CKA, не по минимуму QLIKE; S-CD, S-C7, S-CKL не кандидаты;
пороги У1-S 0,415 / У2-S 0,50 / У3-S 1,00 — константы (граничные значения: 0,415 проходит, 0,4151 нет); У4-C — строгое неравенство с S-CKL и S-C7; Σ вкладов лет = Δ; STRUCT = К1 ∧ К2 ∧ К3.
Синтетика на реальных датах t⁺ окна D_M (маски периодов W5), мир с известным ответом: S-CK и S-CKA — наилучшие, контроли хуже. Реальные файлы не пишутся."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s18 import cm_eval_chk as C
from models_v2.s18 import cm_eval_tab as T
from models_v2.s18 import cm_eval_yr as Y


def tp_dm() -> pd.DatetimeIndex:
    from models_v2.s16 import x1_w8_diag_core as K
    return pd.DatetimeIndex(K.load()["t_plus"])


def world(sd: dict, seed: int = 5):
    tp = tp_dm()
    n = len(tp)
    rng = np.random.default_rng(seed)
    ft = np.exp(rng.normal(-1.0, 0.4, n))
    rv = ft * rng.chisquare(3, n) / 3
    F = {k: ft * np.exp(s * rng.normal(size=n)) for k, s in sd.items()}
    F.update({"X1_C(%s)" % v: F[v] * np.exp(0.01 * rng.normal(size=n)) for v in T.VARS})
    return rv, tp, F


SD_BEST = {"K1": 0.5, "K2": 0.45, "K3": 0.45, "X1": 0.4, "W5-G": 0.4, "W8-G": 0.4, "B_mix": 0.4, "ZB": 0.4, "S-A2": 0.4, "S-CD": 0.35, "S-CK": 0.05, "S-CKA": 0.02, "S-C7": 0.4, "S-CKL": 0.35}


@pytest.fixture(scope="module")
def best():
    rv, tp, F = world(SD_BEST)
    ls = T.losses(rv, F)
    tbl, yrs, cnc = T.table(rv, tp, F), Y.years(tp, ls), Y.conc(tp, ls)
    st = T.stats(rv, tp, F, ls, T.spa_blocks(ls), T.struct_spa(ls))
    return dict(tp=tp, ls=ls, tbl=tbl, yrs=yrs, cnc=cnc, st=st, g=Y.guard(tbl, yrs, cnc, st))


def test_constants_and_order():
    assert Y.THR == {"У1-S": 0.415, "У2-S": 0.50, "У3-S": 1.00} and Y.ORDER == ("S-CK", "S-CKA") and Y.CTRL == ("S-CKL", "S-C7")
    assert "S-CD" not in Y.ORDER and "S-C7" not in Y.ORDER and "S-CKL" not in Y.ORDER


def test_identity_sum_of_period_contributions_equals_delta(best):
    assert Y.identity_gap(best["yrs"]) <= 1e-12
    y = best["yrs"]
    assert set(zip(y["series"], y["base"])) >= {("S-CK", "S-CD"), ("S-CK", "S-C7"), ("S-CK", "S-CKL"), ("S-CKA", "S-CK"), ("X1_C(S-CK)", "X1"), ("S-CK", "W5-G")}


def test_world_with_known_answer_gives_new_control_s_ck_not_min_qlike(best):
    g, q = best["g"], best["tbl"].set_index("series")["qlike"]
    assert q["S-CKA"] < q["S-CK"] < q["S-CD"]                                # S-CKA численно лучше, но порядок S-CK → S-CKA
    d = g[g["id"] == "решение"]["value"].iloc[0]
    assert d.startswith("новая контрольная версия: S-CK (") and "S-CKA" not in d.split("(")[0] and "S-CD" not in d
    assert Y.decide(g) == d
    sub = g[(g["candidate"] == "S-CK") & ~g["id"].isin(Y.SKIP)]
    assert len(sub) == 10 and sub["passed"].all()
    assert g[g["id"] == "решение STRUCT"]["value"].iloc[0].startswith("эффект структуры: да")


def test_mutation_candidate_by_min_qlike_is_caught(best, monkeypatch):
    monkeypatch.setattr(Y, "ORDER", ("S-CKA", "S-CK"))                       # порядок «по минимуму QLIKE»
    assert Y.decide(best["g"]).startswith("новая контрольная версия: S-CKA")
    assert Y.decide(best["g"]) != best["g"][best["g"]["id"] == "решение"]["value"].iloc[0]
    assert C.order_selftest() != []                                           # самотест порядка падает на мутации


def test_mutation_s_cd_declared_candidate_is_caught(best, monkeypatch):
    monkeypatch.setattr(Y, "ORDER", ("S-CD", "S-CK", "S-CKA"))
    with pytest.raises(ValueError):
        Y.decide(best["g"])                                                   # условий S-CD в GUARD нет: кандидатом не является
    assert not any(r.candidate == "S-CD" for r in best["g"].itertuples())
    with pytest.raises(ValueError):
        C.order_selftest()                                                    # самотест порядка на синтетике тоже не принимает S-CD


def test_ctrl_u4c_requires_strict_win_over_s_ckl_and_s_c7():
    SD = dict(SD_BEST, **{"S-CKL": 0.02, "S-CK": 0.05})                      # контроль (б) лучше кандидата
    rv, tp, F = world(SD)
    ls = T.losses(rv, F)
    tbl, yrs, cnc = T.table(rv, tp, F), Y.years(tp, ls), Y.conc(tp, ls)
    st = T.stats(rv, tp, F, ls, T.spa_blocks(ls), T.struct_spa(ls))
    g = Y.guard(tbl, yrs, cnc, st)
    r = g[(g["candidate"] == "S-CK") & (g["id"] == "У4-C")].iloc[0]
    assert not bool(r["passed"]) and "S-CKL" in r["condition"] and "S-C7" in r["condition"]
    d = g[g["id"] == "решение"]["value"].iloc[0]
    assert "У4-C" in d.split("S-CKA")[0] and not d.startswith("новая контрольная версия: S-CK ")
    s = g[g["candidate"] == "структура"].set_index("id")["passed"]
    assert bool(s["К2"]) and not bool(s["К3"]) and not bool(s["STRUCT"])     # К3 не выполнена ⇒ STRUCT нет
    assert g[g["id"] == "решение STRUCT"]["value"].iloc[0].startswith("эффект структуры: нет (не выполнено: К3)")


def frames(d1, d2, d3, tot, loyo, best="2018"):
    """Минимальные YEARS / CONC для `_guard_c`: пара М1 — (X, B), У5-S — (S-CK, W5-G); known answer по порогам."""
    rows = []
    for s, b in (("X", "B"), ("S-CK", "W5-G")):
        rows += [dict(series=s, base=b, period=p, kind="period", delta_pct=(d3 if p == "2018" else 0.0), contrib_pct=(-1.0 if p == best else 0.0), delta_total_pct=tot) for p in ("2018", "2019")]
        rows += [dict(series=s, base=b, period="h1_2018-2021", kind="half", delta_pct=d1, contrib_pct=0.0, delta_total_pct=tot), dict(series=s, base=b, period="h2_2022-2026", kind="half", delta_pct=d2, contrib_pct=0.0, delta_total_pct=tot)]
    return pd.DataFrame(rows), pd.DataFrame([dict(series="S-CK", base="W5-G", item="loyo", key=best, value=loyo)])


def guard_rows(monkeypatch, d1, d2, d3, loyo, q=None):
    monkeypatch.setattr(Y, "hkey", lambda c, b: "X")
    monkeypatch.setattr(Y, "PAIRS", (("М1", "B"),))
    y, cnc = frames(d1, d2, d3, -0.1, loyo)
    rows = []
    Y._guard_c(rows, "S-CK", q or {"S-CK": 1.0, "S-CKL": 2.0, "S-C7": 3.0}, y, cnc)
    return {r["id"]: r for r in rows}


@pytest.mark.parametrize("idn,thr,idx", [("У1-S", 0.415, 0), ("У2-S", 0.50, 1), ("У3-S", 1.00, 2)])
def test_thresholds_are_fixed_constants_at_boundary(monkeypatch, idn, thr, idx):
    for delta, ok in ((thr, True), (thr + 1e-4, False), (-3.0, True)):
        d = [0.0, 0.0, 0.0]
        d[idx] = delta
        r = guard_rows(monkeypatch, *d, 0.0)[idn]
        assert r["passed"] is ok and r["threshold"] == thr                   # порог — константа спецификации, не из результата


def test_u5s_uses_loyo_without_best_period(monkeypatch):
    for loyo, ok in ((0.0, True), (0.0001, False), (-2.0, True)):
        u5 = guard_rows(monkeypatch, 0.0, 0.0, 0.0, loyo)["У5-S"]
        assert u5["passed"] is ok and u5["value"] == loyo and "2018" in u5["condition"]


def test_u4c_needs_strict_inequality_with_both_controls(monkeypatch):
    for q, ok in (({"S-CK": 1.0, "S-CKL": 2.0, "S-C7": 3.0}, True), ({"S-CK": 1.0, "S-CKL": 1.0, "S-C7": 3.0}, False), ({"S-CK": 1.0, "S-CKL": 2.0, "S-C7": 0.9}, False)):
        assert guard_rows(monkeypatch, 0.0, 0.0, 0.0, 0.0, q)["У4-C"]["passed"] is ok


def test_decide_known_answer_with_synthetic_rows_and_struct():
    assert C.order_selftest() == [] and C.struct_selftest() == []


def test_mutation_decide_reversed_order_is_caught_by_selftest(monkeypatch):
    real = Y.decide

    def mut(g):
        monkeypatch.setattr(Y, "ORDER", tuple(reversed(Y.ORDER)))
        try:
            return real(g)
        finally:
            monkeypatch.setattr(Y, "ORDER", ("S-CK", "S-CKA"))
    monkeypatch.setattr(Y, "decide", mut)
    assert any("порядок" in b for b in C.order_selftest())


def test_mutation_struct_without_k1_is_caught(monkeypatch):
    def mut(g):
        s = g[g["candidate"] == Y.SD].set_index("id")["passed"].map(Y.Y._b)
        return "эффект структуры: да (К1 ∧ К2 ∧ К3); подтверждён" if s["К2"] and s["К3"] else "эффект структуры: нет"
    monkeypatch.setattr(Y, "decide_struct", mut)
    assert C.struct_selftest() != []


def test_struct_confirmed_needs_p_holm_and_p_spa_from_k1(best):
    s = best["g"][best["g"]["candidate"] == "структура"].set_index("id")
    k1 = best["st"][(best["st"]["kind"] == "struct") & (best["st"]["hypothesis"] == "К1")].iloc[0]
    want = bool(k1["delta_pct"] < 0 and k1["p_holm_N"] < 0.05 and k1["p_spa_c"] < 0.05)
    assert bool(s.at["STRUCT подтверждён", "passed"]) is want and bool(s.at["STRUCT", "passed"])


def test_power_table_pairs_and_windows(best):
    p = Y.power_table(best["tp"], best["ls"])
    assert len(p) == 10 and set(p["pair"]) == {"М1", "М2", "К1"} and set(p["window"]) == {"D_M", "без 2022"}
    assert (p[p["window"] == "D_M"]["lag"] == 8).all() and (p[p["window"] == "D_M"]["n"] == 2198).all()
    k1 = p[(p["pair"] == "К1") & (p["window"] == "D_M")].iloc[0]
    assert k1["effect_abs"] < 0 and k1["n_p05"] > 0 and k1["n_p08"] > k1["n_p05"]
