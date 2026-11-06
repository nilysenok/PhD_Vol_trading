"""Ж-212: QLIKE-согласованная подгонка. Производные 1 − RV/f и RV/f проверяются численно; положительность f; подгонка восстанавливает известный γ."""
import numpy as np
import pytest

from models_v2 import metrics
from models_v2.s8 import hs_qfit


def num_check(fn, rv, base, g, h=1e-6):
    L, gr, hs = fn(rv, base, g)
    lp, gp, _ = fn(rv, base, g + h)
    lm, gm, _ = fn(rv, base, g - h)
    return np.max(np.abs((lp - lm) / (2 * h) - gr)), np.max(np.abs((gp - gm) / (2 * h) - hs))


def sample(n=50, seed=3):
    rng = np.random.default_rng(seed)
    return np.exp(rng.normal(size=n)), np.exp(rng.normal(size=n)), rng.normal(scale=0.7, size=n)


def test_gradient_and_hessian_numeric():
    rv, base, g = sample()
    eg, eh = num_check(hs_qfit.obj, rv, base, g)
    assert eg < 1e-6 and eh < 1e-6


def test_mutation_wrong_gradient_sign_and_hessian_caught():
    rv, base, g = sample()

    def bad_grad(a, b, c):
        L, gr, h = hs_qfit.obj(a, b, c)
        return L, -gr, h

    def bad_hess(a, b, c):
        L, gr, h = hs_qfit.obj(a, b, c)
        return L, gr, np.ones_like(h)

    assert num_check(bad_grad, rv, base, g)[0] > 1e-3
    assert num_check(bad_hess, rv, base, g)[1] > 1e-3


def test_hessian_positive_and_f_positive_even_for_extreme_g():
    rv, base, _ = sample()
    for gv in (-40.0, 0.0, 40.0):
        L, gr, h = hs_qfit.obj(rv, base, np.full(len(rv), gv))
        assert np.isfinite(L).all() and (h > 0).all()
        assert (base * np.exp(gv) > 0).all()


def test_loss_equals_qlike_up_to_constant():
    rv, base, g = sample()
    L = hs_qfit.obj(rv, base, g)[0]
    q = metrics.qlike(rv, base * np.exp(g))
    assert np.allclose(L - q, np.log(rv) + 1)                    # L = ln f + RV/f = QLIKE + ln RV + 1


def test_fit_recovers_known_gamma():
    rng = np.random.default_rng(4)
    n = 40000
    base = np.exp(rng.normal(size=n) * 0.3)
    x = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    f = base * np.exp(0.2 + 0.5 * x)
    rv = f * rng.normal(size=n) ** 2                              # E[RV] = f
    gam = hs_qfit.fit(X, rv, np.log(base))
    assert gam == pytest.approx([0.2, 0.5], abs=0.04)


def test_fit_zero_when_base_is_truth_and_min_is_stationary():
    rng = np.random.default_rng(5)
    n = 30000
    base = np.exp(rng.normal(size=n) * 0.3)
    X = rng.normal(size=(n, 1))
    rv = base * rng.normal(size=n) ** 2
    gam = hs_qfit.fit(X, rv, np.log(base))
    assert abs(gam[0]) < 0.03
    r = rv / (base * np.exp(hs_qfit.lin(X, gam)))
    assert abs(np.mean(X[:, 0] * (1 - r))) < 1e-8             # условие первого порядка


def test_fit_rejects_empty_and_nonpositive_rv():
    with pytest.raises(ValueError):
        hs_qfit.fit(np.zeros((0, 1)), np.zeros(0), np.zeros(0))
    with pytest.raises(ValueError):
        hs_qfit.fit(np.ones((3, 1)), np.array([1.0, 0.0, 2.0]), np.zeros(3))
