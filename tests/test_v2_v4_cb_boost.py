"""V4-C лист C-b (G19): C7 — бустинг гейта (известный ответ, порог листа, годовой цикл) и градиент C10 против конечных разностей, наклоны по бумагам восстанавливаются."""
import numpy as np
from scipy.optimize import check_grad

from models_v2.s11 import v4_cb_boost as BO, v4_cb_fit as FB


def test_c10_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    n, k, N = 3, 2, 400
    ell, lH, Z, tid = rng.normal(0, 0.4, N), rng.normal(0, 0.3, N), rng.normal(0, 1, (N, k)), rng.integers(0, n, N)
    rv = np.exp(lH + 0.5 * ell) * rng.gamma(30.0, 1 / 30.0, N)
    f = FB._obj_eta(rv, lH, ell, Z, tid, n, 2.0, 3.0)
    x = rng.normal(0, 0.3, n + 1 + k + n * k)
    assert check_grad(lambda v: f(v)[0], lambda v: f(v)[1], x) < 1e-6


def test_c10_per_ticker_slopes_recovered():
    rng = np.random.default_rng(2)
    n, N = 2, 40000
    tid = rng.integers(0, n, N)
    ell, lH, z = rng.uniform(-0.8, 0.8, N), rng.normal(0, 0.4, N), rng.normal(0, 1, N)
    slope = np.array([1.5, -1.5])[tid]
    w = 1 / (1 + np.exp(-slope * z))
    rv = np.exp(lH + w * ell) * rng.gamma(60.0, 1 / 60.0, N)
    r = FB.fit_lin(rv, lH, ell, z[:, None], tid, n, eta=True, kappa=1e-6, lam=1e-6)
    assert r["eta"][0, 0] - r["eta"][1, 0] > 1.5 and r["eta"][0, 0] > 0 > r["eta"][1, 0]      # наклоны бумаг расходятся в верную сторону (истинная разность 3)


def test_c7_known_gate_recovered_by_trees():
    rng = np.random.default_rng(4)
    N = 8000
    X = rng.normal(0, 1, (N, 3))
    ell, lH = rng.uniform(-0.8, 0.8, N), rng.normal(0, 0.4, N)
    w = 1 / (1 + np.exp(-2.0 * X[:, 0]))
    rv = np.exp(lH + w * ell) * rng.gamma(80.0, 1 / 80.0, N)
    m = BO.fit_boost(X, rv, lH, ell, n_trees=100, min_leaf=200)
    eta = BO.predict_eta(m, X)
    assert np.corrcoef(eta, X[:, 0])[0, 1] > 0.8 and BO.n_splits(m) > 100
    assert abs(np.corrcoef(eta, X[:, 1])[0, 1]) < 0.3


def test_c7_min_leaf_500_forbids_split_below_1000_pairs_and_gradient_hessian_known_answer():
    rng = np.random.default_rng(6)
    N = 900
    X = rng.normal(0, 1, (N, 2))
    ell = rng.uniform(-0.8, 0.8, N)
    lH = np.zeros(N)
    m = BO.fit_boost(X, np.exp(0.5 * ell), lH, ell, n_trees=5)
    assert BO.n_splits(m) == 0                                              # 900 < 2·500: ни одно разбиение не оставляет ≥ 500 пар с обеих сторон
    g, h = BO.grad_hess(np.array([2.0]), np.array([0.0]), np.array([0.4]), np.array([0.0]))       # w = 0,5: u = 0,4·0,25 = 0,1; F = exp(0,2); e = 2/F
    e = 2.0 / np.exp(0.2)
    assert np.isclose(g[0], (1 - e) * 0.1) and np.isclose(h[0], e * 0.01)


def test_c7_defaults_are_declared_values():
    assert (BO.N_TREES, BO.DEPTH, BO.SHRINK, BO.MIN_LEAF) == (200, 2, 0.05, 500)
