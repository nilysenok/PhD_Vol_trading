"""B1-6 (Ж-204): правила H3 / R3 / R4, режимы и веса L5 — только прошлое; мутации M-B1-4…8 и № 22."""
import numpy as np
import pytest

from models_v2.s2 import features as ft
from models_v2.s5 import pit
from models_v2.s5.b1 import designs, rules

RNG = np.random.default_rng(3)
N = 700


def world():
    d = np.arange("2019-01-01", N, dtype="datetime64[D]") if False else (np.datetime64("2019-01-01") + np.arange(N)).astype("datetime64[ns]")
    td = np.concatenate([d[1:], [d[-1] + np.timedelta64(1, "D")]])
    rv = np.exp(RNG.normal(0, 0.7, N))
    return d, td, rv, rv * np.exp(RNG.normal(0.1, 0.3, N)), rv * np.exp(RNG.normal(-0.1, 0.5, N))


D, TD, RV, S2H, S2B = world()
GRP = (RNG.uniform(size=N) > 0.7).astype(float)


def corrupt(k):
    rv, h, b = RV.copy(), S2H.copy(), S2B.copy()
    rv[k:], h[k:], b[k:] = 99.0, 0.01, 77.0
    return rv, h, b


@pytest.mark.parametrize("kind", ["exp", "roll", "reg"])
def test_h3_weight_ignores_unclosed_forecasts(kind):
    k = 500
    w0 = rules.h3_weights(kind, RV, S2H, S2B, D, TD, GRP)
    rv, h, b = corrupt(k)
    w1 = rules.h3_weights(kind, rv, h, b, D, TD, GRP)
    assert w0[k] == w1[k] and np.array_equal(w0[:k + 1], w1[:k + 1])


@pytest.mark.parametrize("kind", ["exp", "roll"])
def test_mutation_closed_plus_one_caught(kind):
    k = 500
    rv, h, b = RV.copy(), S2H.copy(), S2B.copy()
    rv[k], h[k], b[k] = 1e-60, 1e6, 1e-60         # пара k: её цель (k+1) на дату k ещё неизвестна; Boost на ней идеален, HAR ужасен
    ok0 = rules.h3_weights(kind, rv, h, b, D, TD)[k] == rules.h3_weights(kind, RV, S2H, S2B, D, TD)[k]
    mut = rules.h3_weights(kind, rv, h, b, D, TD, extra_closed=1)[k] != rules.h3_weights(kind, RV, S2H, S2B, D, TD, extra_closed=1)[k]
    assert ok0 and mut


def test_h3_known_answer_exact_boost_gives_w_zero_and_exact_har_gives_one():
    assert rules.h3_weights("exp", RV, 10 * RV, RV, D, TD)[-1] == 0.0
    assert rules.h3_weights("exp", RV, RV, 10 * RV, D, TD)[-1] == 1.0
    assert rules.h3_weights("exp", RV, RV, RV, D, TD)[-1] == 0.0   # ничья — меньший w
    assert rules.h3_weights("exp", RV, S2H, S2B, D, TD)[100] == 0.5  # < 252 закрытых


def test_theta_uses_only_past_years_and_year_mutation_is_caught():
    ty = np.repeat([2017, 2018, 2019, 2020], 100)
    n = len(ty)
    qw, rv = RNG.integers(0, 5, n).astype(float), np.exp(RNG.normal(0, 0.5, n))
    h, alt = rv * np.exp(RNG.normal(0, 0.6, n)), rv * np.exp(RNG.normal(0, 0.1, n))   # в прошлом alt лучше -> θ = 1
    base = rules.theta_by_year(qw, ty, h, alt, rv)
    alt2 = alt.copy()
    alt2[ty >= 2019] = 1e3 * rv[ty >= 2019]     # будущее относительно θ_2019 (год 2019 и позже): alt ужасен, HAR идеален
    h2 = h.copy()
    h2[ty >= 2019] = rv[ty >= 2019]
    chg = rules.theta_by_year(qw, ty, h2, alt2, rv)
    assert base[2017] == chg[2017] == 5 and base[2018] == chg[2018] and base[2019] == chg[2019]
    mut_a = rules.theta_by_year(qw, ty, h, alt, rv, include_year=True)
    mut_b = rules.theta_by_year(qw, ty, h2, alt2, rv, include_year=True)
    assert mut_a != mut_b                        # M-B1-5: год Y в выборе θ_Y


def test_switch_rule_known_answer():
    qw = np.array([0., 1., 2., 3., 4., np.nan])
    out = rules.switch(qw, np.full(6, 2018), {2018: 3}, np.ones(6), 2 * np.ones(6))
    assert out.tolist() == [1, 1, 1, 2, 2, 1]


def test_logit_recovers_coefficients():
    Z = RNG.normal(size=(20000, 2))
    y = (RNG.uniform(size=20000) < 1 / (1 + np.exp(-(0.3 + 1.2 * Z[:, 0] - 0.7 * Z[:, 1])))).astype(float)
    b = rules.fit_logit(Z, y, lam=1e-3)
    assert np.allclose(b, [0.3, 1.2, -0.7], atol=0.1)


def test_r3_probability_ignores_future_and_label_mutation_is_caught():
    n = 500
    Z = RNG.normal(size=(n, 2))
    g = (RNG.uniform(size=n) < 1 / (1 + np.exp(-Z[:, 0]))).astype(float)
    grid = np.arange(150, 450)
    p0 = rules.r3_prob(Z, g, grid)
    d = 300
    Z2, g2 = Z.copy(), g.copy()
    Z2[d + 1:], g2[d + 1:] = 9.0, 1.0 - g[d + 1:]
    p1 = rules.r3_prob(Z2, g2, grid)
    i = int(np.flatnonzero(grid == d)[0])
    assert np.array_equal(p0[:i + 1], p1[:i + 1])
    g3 = g.copy()
    g3[d + 1] = 1.0 - g[d + 1]                   # метка d+1 неизвестна на d
    assert rules.r3_prob(Z, g3, grid)[i] == p0[i]
    m0, m1 = rules.r3_prob(Z, g, grid, label_shift=0)[i], rules.r3_prob(Z, g3, grid, label_shift=0)[i]
    assert m0 != m1                              # M-B1-6


def fp(rv):
    X = np.full((len(rv), 12), np.nan)
    X[:, 1] = np.log(rv)
    return ft.Fp(np.arange(len(rv)).astype("datetime64[D]"), np.zeros(len(rv), int), rv, X)


def test_regime_columns_are_past_only_and_current_day_mutation_is_caught():
    rv = np.exp(RNG.normal(0, 0.6, 600))
    k = 450
    rv2 = rv.copy()
    rv2[k + 1:] = 123.0
    a, b = designs.regime_cols(fp(rv)), designs.regime_cols(fp(rv2))
    for key in a:
        assert np.array_equal(a[key][:k + 1], b[key][:k + 1], equal_nan=True)
    x, x2 = rv.copy(), rv.copy()
    x2[k] = 55.0
    assert pit.q_prior_nan(x, .8)[k] == pit.q_prior_nan(x2, .8)[k]           # квантиль без текущего дня
    assert pit.past_bucket(x, [.8], include_current=True)[k] != pit.past_bucket(x2, [.8], include_current=True)[k] or \
        pit.q_prior_nan(x2, .8)[k] != np.nanquantile(x2[k - 251:k + 1], .8)   # № 22: с текущим днём квантиль другой


def test_l5_weights_normalised_on_window_pairs_only():
    F = fp(np.exp(RNG.normal(0, 0.6, 400)))
    idx = np.arange(0, 300)
    w0 = designs.weights_l5(F, 0.5, idx)
    F2 = fp(np.exp(RNG.normal(0, 0.6, 400)))
    F2.X[:300] = F.X[:300]
    assert np.array_equal(w0[idx], designs.weights_l5(F2, 0.5, idx)[idx])      # строки > T не влияют
    mut = np.exp(-0.5 * F.X[:, 1]) / np.exp(-0.5 * F.X[:, 1]).mean()
    mut2 = np.exp(-0.5 * F2.X[:, 1]) / np.exp(-0.5 * F2.X[:, 1]).mean()
    assert not np.array_equal(mut[idx], mut2[idx])                              # M-B1-8: нормировка по всей выборке
    assert abs(w0[idx].mean() - 1) < 1e-12
