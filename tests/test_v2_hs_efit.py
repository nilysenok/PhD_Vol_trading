"""Ж-212 (HS8, G11, G14): активность коррекции a_t — производные, известный ответ, обрезка (порча строк > d не меняет a[:d+1]), a ∈ [0, 1], a = 0 до 252 закрытых, «закрыто + 1» ловится."""
from __future__ import annotations

import numpy as np

from models_v2.s8 import hs_efit as E


def grid_dates(n):
    d = np.arange(n).astype("datetime64[D]").astype("datetime64[ns]")
    return d, d + np.timedelta64(1, "D")


def synth(n=900, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, n)
    har = np.exp(rng.normal(0, 0.3, n))
    g = rng.normal(0.5, 0.4, n)
    a_true = (z > 0).astype(float)
    rv = har * np.exp(a_true * g) * rng.gamma(50.0, 1 / 50.0, n)
    return z[:, None], rv, har, g


def test_loss_gradient_matches_finite_differences():
    Z, rv, har, g = synth(300)
    Zc = np.column_stack([np.ones(300), Z])
    th = np.array([0.3, -0.7])
    L0, gr = E.loss_grad(th, Zc, rv, har, g, 0.01)
    for j in range(2):
        e = np.zeros(2)
        e[j] = 1e-6
        num = (E.loss_grad(th + e, Zc, rv, har, g, 0.01)[0] - E.loss_grad(th - e, Zc, rv, har, g, 0.01)[0]) / 2e-6
        assert abs(num - gr[j]) < 1e-6


def test_known_answer_activity_follows_indicator_in_unit_interval_and_zero_before_252():
    Z, rv, har, g = synth()
    d, td = grid_dates(len(rv))
    a = E.activity(Z, rv, har, g, d, td, ridge=0.01)
    assert (a[:252] == 0).all() and ((a >= 0) & (a <= 1)).all()
    late = slice(600, None)
    assert a[late][Z[late, 0] > 0].mean() > a[late][Z[late, 0] <= 0].mean() + 0.3      # a выше там, где коррекция помогает


def test_truncation_corrupted_future_does_not_change_past_activity():
    Z, rv, har, g = synth()
    d, td = grid_dates(len(rv))
    a = E.activity(Z, rv, har, g, d, td)
    k = 500
    m = np.ones(len(rv))
    m[k + 1:] = np.random.default_rng(3).lognormal(0, 1, len(rv) - k - 1)
    ac = E.activity(Z * m[:, None], rv * m, har * m, g * m, d, td)
    assert np.array_equal(a[:k + 1], ac[:k + 1])
    assert not np.array_equal(a[k + 1:], ac[k + 1:])


def test_mutation_closed_plus_one_leaks_the_future():
    Z, rv, har, g = synth()
    d, td = grid_dates(len(rv))
    k = 500
    m = np.ones(len(rv))
    m[k:] = np.random.default_rng(3).lognormal(0, 1, len(rv) - k)      # портим только цель rv с дня k (RV d_k+1 — будущее на d_k)
    ok1, ok2 = E.activity(Z, rv, har, g, d, td), E.activity(Z, rv * m, har, g, d, td)
    a1, a2 = E.activity(Z, rv, har, g, d, td, extra=1), E.activity(Z, rv * m, har, g, d, td, extra=1)
    assert np.array_equal(ok1[:k + 1], ok2[:k + 1])                    # правильный код: порча цели не просачивается
    assert not np.array_equal(a1[:k + 1], a2[:k + 1])                  # с подложенным «закрыто + 1» просачивается — тест ловит


def test_expanding_median_ignores_future():
    x = np.log(np.random.default_rng(1).lognormal(0, 1, 300))
    y = x.copy()
    y[200:] += 50
    assert np.array_equal(E.expanding_dev(x)[:200], E.expanding_dev(y)[:200], equal_nan=True)
