"""S-D-EVAL: известные ответы таблиц и правил (QLIKE на всех датах, Δ и знак, вердикты разд. 5, сводный вывод, вклады лет, срезы, обрезка) — на синтетике (E.G3)."""
import math

import numpy as np
import pandas as pd

from models_v2.s18 import d_eval_tab as T

N = 2198
TP = pd.bdate_range("2018-01-04", periods=N)


def syn(seed=0, gain=0.0):
    rng = np.random.default_rng(seed)
    rv = np.exp(rng.normal(size=N))
    fb = rv * np.exp(0.5 * rng.normal(size=N))
    return rv, {"b": fb, "h": fb * np.exp(-gain)}


def test_qlike_known_answers_and_window_is_mean_of_days_not_of_years():
    assert T.ql([1.0], [1.0])[0] == 0.0 and abs(T.ql([2.0], [1.0])[0] - (2.0 - math.log(2.0) - 1.0)) < 1e-15
    rv, F = syn()
    t = T.table(rv, TP, {**F, "S-RCK": F["b"], "S-RCKm": F["h"], "S-RCKf": F["h"], "S-RCKb": F["b"], "W5-G": F["b"], "X1": F["b"], "K2": F["b"], "K3": F["b"]})
    q = T.ql(rv, F["b"])
    y = TP.year.to_numpy()
    by_year = float(np.mean([q[y == k].mean() for k in sorted(set(y))]))
    row = t[t.series == "S-RCK"].iloc[0]
    assert abs(row.qlike - q.mean()) < 1e-15 and abs(row.qlike - by_year) > 1e-6 and row.n == N and row.first_tplus == "2018-01-04"


def test_delta_sign_base_and_interval_known_answer():
    rv, F = syn(1, gain=0.02)
    ls = T.losses(rv, F)
    d = T.deltas(ls, TP, [("h", "b"), ("b", "h")]).set_index("series")
    want = 100.0 * (ls["h"].mean() - ls["b"].mean()) / ls["b"].mean()
    assert abs(d.at["h", "delta_pct"] - want) < 1e-12 and d.at["h", "delta_pct"] * d.at["b", "delta_pct"] < 0
    perfect = T.deltas(T.losses(rv, {"p": rv, "b": F["b"]}), TP, [("p", "b")]).iloc[0]
    assert abs(perfect.delta_pct + 100.0) < 1e-12 and perfect.hi90 < 0    # F = RV: QLIKE 0, Δ = −100 %, интервал строго ниже нуля
    assert d.at["h", "lo90"] <= d.at["h", "delta_pct"] <= d.at["h", "hi90"] and (d.boot_B == 10000).all() and (d.boot_block == 20).all() and (d.boot_seed == 0).all() and (d.nw_lag == 8).all()
    assert abs(d.at["b", "delta_pct"] - 100.0 * (ls["b"].mean() - ls["h"].mean()) / ls["h"].mean()) < 1e-12 and abs(d.at["b", "delta_pct"] + want) > 1e-3


def test_verdict_four_branches_and_boundaries():
    assert T.verdict(0.3, 0.1, 0.5) == "needed" and T.verdict(0.3, -0.1, 0.5) == "worse_noise" and T.verdict(0.3, 0.0, 0.5) == "worse_noise"
    assert T.verdict(-0.3, -0.5, 0.1) == "not_found" and T.verdict(-0.3, -0.5, -0.1) == "harms" and T.verdict(-0.3, -0.5, 0.0) == "not_found" and T.verdict(0.0, -0.1, 0.1) == "not_found"
    assert "хуже" in T.VERDICT["needed"] and "хуже" in T.VERDICT["worse_noise"] and "лучше" in T.VERDICT["harms"]


def test_summary_rules_per_spec_section_5():
    s = T.summary
    assert s({"i": "needed", "iii": "needed"}) == T.SUMMARY["both_needed"]
    assert s({"i": "not_found", "iii": "not_found"}) == T.SUMMARY["i_not_found"] and s({"i": "needed", "iii": "harms"}) == T.SUMMARY["iii_harms"]
    assert T.SUMMARY["i_not_found"] in s({"i": "not_found", "iii": "harms"}) and T.SUMMARY["iii_harms"] in s({"i": "not_found", "iii": "harms"})
    assert "не предусмотрено" in s({"i": "worse_noise", "iii": "not_found"}) and "не предусмотрено" in s({"i": "needed", "iii": "worse_noise"})


def test_years_sum_equals_delta_and_slices_denominator_is_base_on_kept_dates():
    rv, F = syn(2, gain=0.03)
    ls = T.losses(rv, F)
    y = T.years(TP, ls, [("h", "b")])
    assert abs(y.contrib_pct.sum() - y.delta_total_pct.iloc[0]) < 1e-12 and len(y) == len(set(TP.year))
    s = T.slices(TP, ls, [("h", "b")]).set_index("slice")
    for nm, n in ((T.SLICES[0], N), (T.SLICES[1], N - 1), (T.SLICES[2], int((TP.year != 2022).sum())), (T.SLICES[3], N - 44)):
        assert s.at[nm, "n"] == n and "знаменатель" in s.at[nm, "control_and_denominator"]
    m = TP != pd.Timestamp("2022-02-24")
    assert abs(s.at[T.SLICES[1], "delta_pct"] - 100.0 * (ls["h"][m] - ls["b"][m]).sum() / ls["b"][m].sum()) < 1e-12
    assert s.at[T.SLICES[0], "n_removed"] == 0 and s.at[T.SLICES[3], "n_removed"] == 44


def test_trim_removes_22_best_and_22_worst_by_difference_and_sets_differ_between_pairs():
    d = np.zeros(N)
    d[:5], d[-5:] = -10.0, 10.0
    keep = T.trim_keep(d)
    assert (~keep).sum() == 44 and not keep[:5].any() and not keep[-5:].any()
    rv, F = syn(3, gain=0.05)
    ls = T.losses(rv, {**F, "g": F["b"] * np.exp(0.01 * np.sin(np.arange(N)))})
    ov = T.trim_overlap(TP, ls, ("h", "b"), ("g", "b"))
    assert ov["n_each"] == 44 and 0 <= ov["n_common"] < 44


def test_spec_constants_are_not_taken_from_results():
    assert T.THR_PM == -0.0313 and T.THR_PF == 0.0157 and T.KTRIM == 22 == math.ceil(0.01 * N) and T.SHARE_SPEC == 0.89
