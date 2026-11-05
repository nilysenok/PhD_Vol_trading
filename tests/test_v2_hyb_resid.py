"""GB1: Б-1 (остаток к HAR-J(б)) и обобщённый запуск gen.run_design на синтетике; мутация «остаток по HAR на всей выборке»."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models_v2 import har  # noqa: E402
from models_v2.s2 import boost  # noqa: E402
from models_v2.s2 import features as ft  # noqa: E402
from models_v2.s3 import gen, resid  # noqa: E402


def panel(n=700, seed=2):
    rng = np.random.default_rng(seed)
    lv = np.zeros(n)
    for i in range(1, n):
        lv[i] = 0.9 * lv[i - 1] + rng.normal(0, 0.4)
    rv = np.exp(lv)
    s = pd.Series(rv, index=pd.bdate_range("2016-01-04", periods=n))
    df = pd.DataFrame({"rv": s, "rv_w": s.rolling(5, min_periods=5).mean(), "rv_m": s.rolling(22, min_periods=22).mean(),
                       "j": rng.exponential(0.2, n) * s, "seg": 0})
    return har.make_panel(df.iloc[21:])


def test_lb_equals_stage2_feature_and_a_below_b():
    p = panel()
    la, lb = resid.harj_ab(p)
    ref = ft.har_feature(p)
    ok = np.isfinite(ref)
    assert ok.sum() > 100 and np.allclose(lb[ok], ref[ok], rtol=0, atol=1e-12) and (la[ok] < lb[ok]).all()


def test_lb_is_pit_and_full_sample_mutation_caught():
    p = panel()
    t = 400
    base = resid.harj_ab(p)[1][t]
    q = har.Panel(p.dates, p.seg, p.rv.copy(), p.X.copy())
    q.rv[t + 1:] *= 5
    q.X[t + 1:] += 1.0
    assert resid.harj_ab(q)[1][t].tobytes() == base.tobytes()
    assert resid.harj_ab(q, full=True)[1][t] != resid.harj_ab(p, full=True)[1][t]  # мутация видна на будущем


def test_target_minus_segment_break_and_value():
    F = ft.Fp(np.arange(5).astype("datetime64[D]"), np.array([0, 0, 1, 1, 1]), np.e ** np.arange(5.0), np.zeros((5, 12)))
    y = resid.target_minus(F, np.full(5, 0.5))
    assert y[0] == pytest.approx(0.5) and np.isnan(y[1]) and y[2] == pytest.approx(2.5) and np.isnan(y[4])


def design(n=800, seed=4, noise=0.3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 12))
    y = np.where(X[:, 0] > 0, 0.8, -0.8) + rng.normal(0, noise, n)
    return gen.Design(X, np.zeros(n, int), y, tuple(range(11)))


def test_run_design_recovers_known_residual_and_s2():
    D = design()
    spec = boost.Spec("xgb", "mse", False, False)
    r = gen.run_design(D, spec, 3, 0.5, 700, np.arange(701, 799))
    truth = np.where(D.X[701:799, 0] > 0, 0.8, -0.8)
    assert np.corrcoef(r["f"], truth)[0, 1] > 0.9 and 0.04 < r["s2_es"] < 0.16


def test_run_design_future_targets_do_not_change_forecast():
    D = design()
    spec = boost.Spec("lgbm", "mse", False, False)
    a = gen.run_design(D, spec, 3, 0.5, 700, [700])["f"]
    y2 = D.y.copy()
    y2[700:] += 9.0
    D2 = gen.Design(D.X, D.seg, y2, D.cols)
    assert gen.run_design(D2, spec, 3, 0.5, 700, [700])["f"].tobytes() == a.tobytes()


def test_run_design_oracle_improves_and_perm_does_not():
    D = design(noise=0.6)
    spec = boost.Spec("xgb", "mse", False, False)
    d = np.arange(701, 799)
    mse = lambda v: float(np.mean((gen.run_design(D, spec, 3, 0.5, 700, d, v)["f"] - D.y[d]) ** 2))  # noqa: E731
    assert mse("oracle") < 0.5 * mse("work") and mse("perm") >= mse("work")


def test_sigma2_b1_scales_with_c_and_bias():
    h, f, s2 = np.zeros(3), np.array([1.0, 0.0, -1.0]), 0.4
    assert resid.sigma2_b1(h, f, 0.5, s2).tolist() == pytest.approx(np.exp(0.5 * f + 0.2).tolist())
    assert resid.sigma2_b1(h, f, 1.0, s2, bias=False).tolist() == pytest.approx(np.exp(f).tolist())


def test_linear_tree_extrapolates_where_plain_tree_does_not():
    rng = np.random.default_rng(1)
    n = 700
    x0 = np.r_[rng.uniform(0, 1, 600), rng.uniform(2, 3, 100)]
    X = np.column_stack([x0, rng.normal(size=(n, 11))])
    y = 2.0 * x0 + rng.normal(0, 0.05, n)
    D = gen.Design(X, np.zeros(n, int), y, tuple(range(11)), linear=True)
    f_lin = gen.run_design(D, boost.Spec("lgbm", "mse", False, False), 3, 0.5, 599, np.arange(600, 650))["f"]
    Dp = gen.Design(X, D.seg, y, D.cols)
    f_pl = gen.run_design(Dp, boost.Spec("lgbm", "mse", False, False), 3, 0.5, 599, np.arange(600, 650))["f"]
    e_lin, e_pl = np.abs(f_lin - y[600:650]).mean(), np.abs(f_pl - y[600:650]).mean()
    # плоское дерево не выходит за диапазон обучения (≈2); линейные листья выходят, но частично (ридж, lr 0,03)
    assert f_pl.max() <= 2.05 and f_lin.mean() > f_pl.max() + 0.1 and e_lin < e_pl
