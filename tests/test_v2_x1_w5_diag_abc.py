"""W5-A, блоки A–C и D `W5_DIAG.csv` (`x1_w5_diag_abc`, `x1_w5_diag_d`): известный ответ на синтетике — множитель группы c_G = mean(RV/X1), перенос между половинами
(оценка на одной, счёт на другой), π группы, тождество вкладов, смещение в днях события, выигрыш группового множителя. Только синтетика, реальных файлов нет."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s16 import x1_w5_diag_abc as A
from models_v2.s16 import x1_w5_diag_d as D


def ql(rv, f):
    r = np.asarray(rv, float) / np.asarray(f, float)
    return r - np.log(r) - 1.0


def frame(rv, F=1.0, n=240, M=1.0, C=1.0, K1=1.0, K2=1.0):
    """n строк: 8 лет по 30 строк (2018…2025), h1 = 2018–2021. rv, F, M, C, K1, K2 — скаляры или массивы."""
    yr = (2018 + np.arange(n) // (n // 8)).astype(int)
    x = pd.DataFrame(dict(rv=np.broadcast_to(np.asarray(rv, float), n).copy(), F=np.broadcast_to(np.asarray(F, float), n).copy()))
    for k, v in (("M", M), ("C", C), ("K1", K1), ("K2", K2)):
        x[k] = np.broadcast_to(np.asarray(v, float), n).copy()
    x["yr"] = yr.astype(str)
    x["half"] = np.where(yr <= 2021, "h1", "h2")
    x["l_X1"], x["l_K2"], x["l_M"] = ql(x["rv"], x["F"]), ql(x["rv"], x["K2"]), ql(x["rv"], x["M"])
    return x


def rowsof(rows, **kw):
    out = [r for r in rows if all(r[k] == v for k, v in kw.items())]
    assert len(out) == 1, (kw, len(out))
    return out[0]


def ab(n=240):
    return np.where(np.arange(n) % 2 == 0, "a", "b").astype(object)


def test_level_multiplier_transfer_known_answer() -> None:
    n = 240
    g = ab(n)
    h1 = (2018 + np.arange(n) // 30) <= 2021
    rv = np.where(h1, np.where(g == "a", 1.0, 3.0), np.where(g == "a", 2.0, 4.0))             # c_G: h1 {a: 1, b: 3}, c0 = 2; h2 {a: 2, b: 4}, c0 = 3
    x = frame(rv)
    rows = A.block_b(x, {"s": g})
    r = rowsof(rows, block="B_level", window="h1->h2")
    f_ctrl, f_st = np.full(120, 2.0), np.where(g[~h1] == "a", 1.0, 3.0)                  # множители h1 применены к h2
    q_x, q_c, q_s = ql(rv[~h1], 1.0).mean(), ql(rv[~h1], f_ctrl).mean(), ql(rv[~h1], f_st).mean()
    assert (r["n"], r["ctrl_param"], r["group"]) == (120, 2.0, "a:1.000|b:3.000")
    assert r["q_X1"] == pytest.approx(q_x, rel=1e-12) and r["q_ctrl"] == pytest.approx(q_c, rel=1e-12) and r["q_state"] == pytest.approx(q_s, rel=1e-12)
    assert r["d_state_vs_X1_pct"] == pytest.approx(100 * (q_s / q_x - 1), rel=1e-12) and r["d_state_vs_ctrl_pct"] == pytest.approx(100 * (q_s / q_c - 1), rel=1e-12)
    assert r["q_state"] > 0                                                               # перенос, а не in-sample: на h2 множители h1 не идеальны
    r2 = rowsof(rows, block="B_level", window="h2->h1")
    assert (r2["ctrl_param"], r2["group"]) == (3.0, "a:2.000|b:4.000")
    for w in ("h1->h1_insample", "h2->h2_insample"):
        ri = rowsof(rows, block="B_level", window=w)
        assert ri["q_state"] == pytest.approx(0.0, abs=1e-15) and ri["d_state_vs_X1_pct"] == pytest.approx(-100.0)    # в своей половине множитель точен
    assert len([r for r in rows if r["block"] == "B_trust"]) == 0                         # состояние `s` не из списка доверия


def test_level_na_rows_excluded_from_fit_and_evaluation() -> None:
    n = 240
    g = ab(n)
    g[:5] = "na"
    rv = np.where(g == "b", 3.0, 1.0)
    rv[:5] = 100.0                                                                        # na-строки с выбросом: в оценку не входят
    r = rowsof(A.block_b(frame(rv), {"s": g}), block="B_level", window="h1->h1_insample")
    assert r["n"] == 120 - 5 and r["ctrl_param"] == pytest.approx(((rv[5:120] / 1.0).mean()))
    assert r["group"] == "a:1.000|b:3.000"


def test_no2022_window_drops_year_2022() -> None:
    g = ab()
    rows = A.block_b(frame(np.where(g == "a", 1.0, 3.0)), {"s": g})
    assert rowsof(rows, block="B_level", window="h1->h2")["n"] == 120
    assert rowsof(rows, block="B_level", window="h1->h2_no2022")["n"] == 120 - 30         # 2022 = 30 строк


def test_trust_pi_group_known_answer() -> None:
    n = 240
    g = ab(n)
    M, Cc = 1.0, 2.0
    rv = np.where(g == "a", 1.0, 2.0)                                                     # a: лучший π = 0 (F = M), b: π = 1 (F = C)
    rows = A.block_b(frame(rv, M=M, C=Cc), {"e": g, "gap": g})
    assert [r["block"] for r in rows if r["state"] == "gap"] and all(r["block"] == "B_level" for r in rows if r["state"] == "gap")      # у gap нет B_trust
    r = rowsof(rows, block="B_trust", state="e", window="h1->h2")
    assert r["group"] == "a:0.000|b:1.000" and r["q_state"] == pytest.approx(0.0, abs=1e-15) and r["d_state_vs_X1_pct"] == pytest.approx(-100.0)
    pc = A.GRID[np.argmin([ql(rv[:120], np.exp(p * np.log(2.0))).mean() for p in A.GRID])]
    assert r["ctrl_param"] == pytest.approx(pc)                                           # контроль — один π на обе группы
    assert len(rows) == 2 * 5 + 5                                                         # e: level + trust, gap: level
    assert A.GRID[0] == 0 and A.GRID[-1] == 1 and len(A.GRID) == 21 and A.MINPAST == 60 and A.PERS == [str(y) for y in range(2018, 2027)]


def test_block_a_contributions_sum_to_total_delta_and_group_means() -> None:
    rng = np.random.default_rng(0)
    n = 240
    rv = rng.lognormal(0, 0.4, n)
    x = frame(rv, F=rng.lognormal(0, 0.1, n), K2=rng.lognormal(0, 0.1, n), M=rng.lognormal(0, 0.1, n))
    g = rng.choice(np.array(["g1", "g2", "g3"], object), n)
    rows = A.block_a(x, {"s": g})
    allr = [r for r in rows if r["window"] == "all"]
    assert sum(r["n"] for r in allr) == n
    delta = 100 * (x["l_X1"].mean() - x["l_K2"].mean()) / x["l_K2"].mean()
    assert sum(r["contrib_K2_pp"] for r in allr) == pytest.approx(delta, rel=1e-10)       # Σ вкладов групп = Δ X1 к K2 (знак: минус — X1 лучше)
    for k in ("g1", "g2", "g3"):
        r = rowsof(rows, state="s", group=k, window="all")
        m = g == k
        assert r["mean_D_K2"] == pytest.approx((x["l_X1"] - x["l_K2"])[m].mean()) and r["mean_D_M"] == pytest.approx((x["l_X1"] - x["l_M"])[m].mean())
        assert r["mean_lnRVF_X1"] == pytest.approx(np.log(x["rv"] / x["F"])[m].mean())


def test_block_a_known_sign_x1_better_gives_negative_contribution() -> None:
    n = 240
    rv = np.full(n, 2.0)
    r = rowsof(A.block_a(frame(rv, F=2.0, K2=1.0), {"s": np.full(n, "z", object)}), window="all")
    assert r["contrib_K2_pp"] < 0 and r["mean_D_K2"] < 0 and r["years_same_sign"] == 8 and r["n_periods"] == 8     # X1 точен, K2 нет


def test_block_c_event_bias_known_answer() -> None:
    n = 240
    ev = np.where(np.arange(n) % 4 == 0, "1", "0").astype(object)
    rv = np.where(ev == "1", 1.5, 1.0)                                                    # X1 = 1: u = ln 1.5 в днях события, 0 в остальных
    x = frame(rv)
    names = ("ev_cbr", "ev_cpius", "ev_fomc", "ev_exp", "ev_rebal", "ev_any", "gap")
    rows = A.block_c(x, {k: ev for k in names})
    assert len(rows) == 7 * 12
    r = rowsof(rows, state="gap", window="all")
    assert r["n"] == 60 and r["bias_diff_X1"] == pytest.approx(np.log(1.5)) and r["bias_diff_K1"] == pytest.approx(np.log(1.5))
    assert r["qlike_ratio_X1"] > 1e10 or np.isinf(r["qlike_ratio_X1"])                    # знаменатель 0 (RV = F вне события)
    assert rowsof(rows, state="ev_any", window="h1")["n"] == 30


def test_gain_known_answer_and_sign() -> None:
    g = np.array(["a", "b"] * 50, object)
    rv = np.where(g == "a", 1.0, 3.0)
    gain, cg = D.gain(rv, np.ones(100), g)
    q_common = ql(rv, np.full(100, 2.0)).mean()
    assert cg == {"a": 1.0, "b": 3.0} and gain == pytest.approx(-100.0) and q_common > 0       # группы точны: потеря 0, общий множитель 2 хуже
    gain0, cg0 = D.gain(np.full(100, 2.0), np.ones(100), g)
    assert gain0 == pytest.approx(0.0, abs=1e-9) or np.isnan(gain0)                       # группы не различаются: выигрыша нет (0/0)
    g1, _ = D.gain(rv, np.ones(100), np.array(["a"] * 100, object))
    assert g1 == pytest.approx(0.0)                                                       # одна группа = общий множитель


def synthetic_gap_frame(n=240):
    i = np.arange(n)
    delta = np.where(i % 5 == 0, 2, np.where(i % 7 == 3, 1, np.where(i % 11 == 4, 4, 0)))
    d = pd.Timestamp("2018-01-01") + pd.to_timedelta(i, unit="D")
    x = frame(np.where(delta > 0, 1.2, 1.0))
    x["d"], x["t_plus"] = d, d + pd.to_timedelta(delta + 1, unit="D")
    x["s"] = (i % 3 == 0).astype(int)
    return x, delta


def test_rows_d_gap_multipliers_delta_categories_and_shift252() -> None:
    x, delta = synthetic_gap_frame()
    gap = np.where(delta > 0, "1", "0").astype(object)                                    # метка = 1[Δ_d > 0], Δ = (t⁺ − d) − 1 календарных дней
    rows = D.rows_d(x, {"gap": gap})
    r = rowsof(rows, state="gap", group="model=X1", window="all")
    assert r["n"] == 240 and r["n_gap"] == int((delta > 0).sum())
    assert r["c_gap"] == pytest.approx(1.2) and r["c_nogap"] == pytest.approx(1.0) and r["ratio_c"] == pytest.approx(1.2)
    assert r["mean_u_diff"] == pytest.approx(np.log(1.2)) and r["gain_vs_common_pct"] < 0
    for k, v in (("0", 0), ("1", 1), ("2", 2), (">=3", 4)):
        assert rowsof(rows, state="delta_cat", group=k, window="all")["n"] == int((delta == v).sum())
    sh = rowsof(rows, state="gap_shift252", window="all")
    assert sh["c_gap"] == pytest.approx(float((np.roll(gap, 252) == gap).mean()))
    assert sh["mean_u_diff"] == pytest.approx(float(((np.roll(gap, 252) == "1") & (gap == "1")).sum() / (gap == "1").sum()))
    assert rowsof(rows, state="gap_x_e", group="e=1,gap=1", window="all")["n"] == int(((delta > 0) & (x["s"].to_numpy() == 1)).sum())


def test_rows_d_stress_drops_exactly_the_largest_one_percent_losses() -> None:
    x, delta = synthetic_gap_frame()
    x.loc[[10, 100, 200], "rv"] = [50.0, 60.0, 70.0]                                      # три различных крупных потери (240 строк: 1 % = 2,4 → 3 строки сверх 99-го процентиля)
    x["l_X1"] = ql(x["rv"], x["F"])
    gap = np.where(delta > 0, "1", "0").astype(object)
    r = rowsof(D.rows_d(x, {"gap": gap}), state="gap", group="model=X1", window="all_no_top1pct")
    assert r["n"] == 237
    assert r["n_gap"] == int((delta > 0).sum()) - 3 and r["c_gap"] == pytest.approx(1.2)  # строки 10, 100, 200 — дни gap (i % 5 == 0): убраны именно они
