"""S-R-EVAL: известные ответы и мутации оценки R на синтетике (Δ_trim, порядок кандидатов, П, Холм N, знак Δ, MCS, NaN-p). Автор тестов — автор кода (по поручению)."""
import numpy as np
import pandas as pd

from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import r_eval_chk as C
from models_v2.s18 import r_eval_tab as T
from models_v2.s18 import r_eval_yr as Y

ORIG_TRIM, ORIG_ORDER, ORIG_OLD = T.trim_delta, Y.ORDER, dict(T.OLD)


def test_honest_selftests_are_clean():
    assert C.trim_selftest() == [] and C.order_selftest() == [] and C.proc_selftest() == []


def test_trim_known_answer_two_sided_k22_on_remaining_dates():
    n = 100
    lb = np.ones(n)
    lh = lb.copy()
    lh[0], lh[1] = -9.0, 11.0                                    # одна самая выгодная и одна самая невыгодная дата
    assert T.trim_delta(lh, lb, k=1) == 0.0                      # обе срезаны, остальные равны
    assert T.trim_delta(lh, lb, k=0) == 100.0 * (-10.0 + 10.0) / n
    assert abs(T.trim_delta(lh - 0.01, lb, k=1) + 1.0) < 1e-12   # все остальные лучше на 1 % базы


def test_share_top_known_answer():
    lb = np.ones(200)
    lh = lb.copy()
    lh[:2] -= 5.0                                                # весь выигрыш — две даты
    assert abs(T.share_top(lh, lb, k=2) - 1.0) < 1e-12 and np.isnan(T.share_top(lb + 0.1, lb, k=2))


def test_mut_trim_one_sided_is_caught(monkeypatch):
    def mut(lh, lb, k=22):
        d = np.asarray(lh, float) - np.asarray(lb, float)
        keep = np.ones(len(d), bool)
        keep[np.argsort(d, kind="stable")[:k]] = False
        return 100.0 * d[keep].sum() / np.asarray(lb, float)[keep].sum()
    monkeypatch.setattr(T, "trim_delta", mut)
    assert C.trim_selftest() != []


def test_mut_trim_base_over_all_dates_is_caught(monkeypatch):
    def mut(lh, lb, k=22):
        d = np.asarray(lh, float) - np.asarray(lb, float)
        keep = np.ones(len(d), bool)
        o = np.argsort(d, kind="stable")
        keep[o[:k]], keep[o[len(d) - k:]] = False, False
        return 100.0 * d[keep].sum() / np.asarray(lb, float).sum()
    monkeypatch.setattr(T, "trim_delta", mut)
    assert C.trim_selftest() != []


def test_mut_k_not_22_is_caught(monkeypatch):
    monkeypatch.setattr(T, "KTRIM", 21)
    assert C.trim_selftest() != []


def test_mut_candidate_order_reversed_or_by_min_qlike_is_caught(monkeypatch):
    monkeypatch.setattr(Y, "ORDER", ("S-RCK", "S-RA2"))
    assert C.order_selftest() != []


def test_mut_procedure_against_w5g_instead_of_old_series_is_caught(monkeypatch):
    monkeypatch.setattr(T, "OLD", {"S-RA2": "W5-G", "S-RCK": "W5-G"})
    assert C.proc_selftest() != []


def test_mut_procedure_without_trim_condition_is_caught(monkeypatch):
    monkeypatch.setattr(T, "trim_delta", lambda lh, lb, k=22: -1.0)
    assert C.proc_selftest() != []


def _stats_rows():
    rng = np.random.default_rng(3)
    n = 2198
    rv = np.exp(rng.normal(0, 0.5, n))
    F = {k: np.exp(rng.normal(0, 0.05, n)) * f for k, f in (("K1", 1.1), ("K2", 1.0), ("K3", 1.0), ("X1", 1.0), ("W5-G", 1.0), ("W8-G", 1.0), ("B_mix", 1.0), ("ZB", 1.0), ("S-A2", 1.0), ("S-CK", 1.0))}
    for v in T.RVARS:
        F[v], F["X1_R(%s)" % v], F["Ffr(%s)" % v] = (np.exp(rng.normal(0, 0.05, n)),) * 3
    F["X1_A(S-A2)"], F["X1_V(S-CK)"] = F["X1"], F["X1"]
    tp = pd.bdate_range("2018-01-04", periods=n)
    ls = T.losses(rv, F)
    return rv, tp, F, ls


def test_mut_holm_n_equals_number_of_rows_is_caught(monkeypatch):
    rv, tp, F, ls = _stats_rows()
    spa, pspa = T.spa_blocks(ls), T.proc_spa(ls)
    assert C._holm_problems(T.stats(rv, tp, F, ls, spa, pspa)) == []
    monkeypatch.setattr(T, "N_HOLM", 8)
    assert C._holm_problems(T.stats(rv, tp, F, ls, spa, pspa)) != []


def test_mut_nw5_in_main_column_is_caught(monkeypatch):
    rv, tp, F, ls = _stats_rows()
    spa, pspa = T.spa_blocks(ls), T.proc_spa(ls)
    monkeypatch.setattr(S, "nw_lag", lambda n: 5)
    assert C._holm_problems(T.stats(rv, tp, F, ls, spa, pspa)) != []


def test_delta_sign_and_base_known_answer():
    rv, tp, F, ls = _stats_rows()
    t = T.table(rv, tp, F).set_index("series")
    q = {k: float(np.mean(S.Q.ql(rv, F[k]))) for k in ("K1", "X1", "K2")}
    assert abs(t.at["K1", "d_X1_pct"] - 100 * (q["K1"] - q["X1"]) / q["X1"]) < 1e-12
    assert abs(t.at["X1", "d_K2_pct"] - 100 * (q["X1"] - q["K2"]) / q["K2"]) < 1e-12 and t.at["X1", "d_K2_pct"] != -t.at["K2", "d_X1_pct"]


def test_verdict_confirmed_only_with_both_p_and_negative_delta():
    assert T.verdict(-0.1, 0.01, 0.01).startswith("численно лучше базы; подтверждено")
    for d, ph, ps in ((-0.1, 0.06, 0.01), (-0.1, 0.01, 0.06), (0.1, 0.01, 0.01)):
        assert "не подтверждено" in T.verdict(d, ph, ps)


def test_nan_p_is_not_turned_into_one_and_mcs_filters_nonpositive():
    p = S.holm_n(np.array([0.001, np.nan, 0.2]), 846)
    assert np.isnan(p[1]) and p[0] < 1.0
    rng = np.random.default_rng(0)
    rv = np.exp(rng.normal(0, 0.3, 300))
    F = {"a": np.ones(300), "b": np.ones(300) * 1.2}
    F["b"][5] = -1.0
    assert int(S.mcs(rv, F)["n_used"].iloc[0]) == 299
