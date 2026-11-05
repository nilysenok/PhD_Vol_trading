"""LGBM волна B: функции потерь L1–L4 (fit_l) на синтетике с известным ответом; custom QLIKE — градиент/гессиан конечными разностями; мутации."""
import numpy as np
import pytest

from models_v2.s4 import fit
from models_v2.s5.b1 import losses
from models_v2.s6.lgbm import fitl, grid
from models_v2.s5.d import lgbfit

HP = grid.Hp(7, 30, 2.0, 0.05, 1.0, 0.8, -1, 120)


def synth(n=700, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 4))
    mu = np.exp(0.5 * X[:, 0])                       # истинное среднее отношения RV/S
    y_ratio = r.gamma(shape=6.0, scale=mu / 6.0)
    return X, y_ratio, np.arange(n - 1)


def test_l4_grad_hess_finite_differences():
    y = np.array([0.3, 1.0, 2.5, 7.0])
    F = np.array([-0.4, 0.1, 0.9, 1.6])
    g, h = losses.grad_hess(y, F)
    gn, hn = losses.numeric_grad_hess(y, F)
    assert np.allclose(g, gn, atol=1e-6) and np.allclose(h, hn, rtol=1e-4, atol=1e-6)


def test_mutation_wrong_sign_gradient_is_caught():
    y, F = np.array([0.3, 2.5]), np.array([-0.4, 0.9])
    g, _ = losses.grad_hess(y, F)
    gn, _ = losses.numeric_grad_hess(y, F)
    assert not np.allclose(-g, gn, atol=1e-6)


@pytest.mark.parametrize("kind", ["lvl", "gamma", "qlike"])
def test_kind_learns_signal_and_is_positive(kind):
    X, yr, idx = synth()
    m = fitl.fit_l(X, yr, idx, HP, kind)
    f = m["predict"](X[:200])
    mu = np.exp(0.5 * X[:200, 0])
    pred = f if kind == "lvl" else np.exp(f)
    assert np.isfinite(pred).all() and m["n_nonfinite_es"] == 0
    assert np.corrcoef(pred, mu)[0, 1] > 0.6                  # сигнал найден
    if kind != "lvl":
        assert (pred > 0).all()


def test_qlike_custom_beats_constant_in_loss():
    X, yr, idx = synth()
    m = fitl.fit_l(X, yr, idx, HP, "qlike")
    f = m["predict"](X)
    const = np.log(yr[idx].mean())
    assert losses.loss(yr, f).mean() < losses.loss(yr, np.full(len(yr), const)).mean()


def test_qlike_init_score_is_in_custom_objective_scores():
    """Известный ответ: y ≈ 3 при шуме признаков — прогноз ≈ ln 3 (если бы preds шли без init_score, получили бы ≈ 2 ln 3)."""
    r = np.random.default_rng(1)
    X = r.normal(size=(600, 3))
    yr = 3.0 * np.exp(0.02 * r.normal(size=600))
    m = fitl.fit_l(X, yr, np.arange(599), HP, "qlike")
    assert abs(float(np.mean(m["predict"](X[:100]))) - np.log(3.0)) < 0.05


def test_qlike_weights_are_applied_in_custom_objective():
    X, yr, idx = synth()
    w = np.where(np.arange(len(yr)) < 350, 1e-6, 1.0)
    a = fitl.fit_l(X, yr, idx, HP, "qlike", w)["predict"](X[:50])
    b = fitl.fit_l(X, yr, idx, HP, "qlike")["predict"](X[:50])
    assert not np.allclose(a, b)


def test_ln_kind_equals_inherited_lgbfit_bitwise():
    class H:    # поля HP XGBoost (fit.HP)
        depth, min_child, l2, lr, subsample, colsample, max_trees = 3, 30.0, 2.0, 0.05, 0.8, 1.0, 120
    X, yr, idx = synth()
    y = np.log(yr)
    a = lgbfit.fit_lgbx(X, y, idx, H)
    b = fitl.fit_l(X, y, idx, grid.from_i2(H), "ln")
    assert a["k"] == b["k"] and a["s2_es"] == b["s2_es"]
    assert np.array_equal(a["predict"](X[:30]), b["predict"](X[:30]))


def test_mutation_lvl_floor_counts_eps_hits():
    s2, n = fitl.sigma2("lvl", np.array([0, 1]), np.array([-1.0, 2.0]), np.nan, np.zeros(2), np.ones(2), np.full(2, 0.1))
    assert n == 1 and (s2 > 0).all() and s2[0] == 0.1


def test_unknown_kind_rejected():
    X, yr, idx = synth(400)
    with pytest.raises(ValueError):
        fitl.fit_l(X, yr, idx, HP, "huber")
    assert fit.PATIENCE == 100
