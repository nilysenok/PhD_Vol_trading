"""Ж-212 (HS9, G12, G14): MoE — геометрическая смесь (мутация: арифметическая), причинный выбор w, Hedge, обрезка, «закрыто + 1», положительность."""
from __future__ import annotations

import numpy as np

from models_v2.s8 import hs_ffit as M


def dates(n):
    d = np.arange(n).astype("datetime64[D]").astype("datetime64[ns]")
    return d, d + np.timedelta64(1, "D")


def synth(n=900, seed=0):
    rng = np.random.default_rng(seed)
    truth = np.exp(rng.normal(0, 0.5, n))
    rv = truth * rng.gamma(40.0, 1 / 40.0, n)
    return rv, truth * np.exp(rng.normal(0, 0.6, n)), truth                 # rv, плохой эксперт (HAR), точный эксперт (ML)


def test_geometric_not_arithmetic():
    a, b = np.array([1.0, 4.0]), np.array([4.0, 16.0])
    assert np.allclose(M.geo(a, b, 0.5), [2.0, 8.0])                         # √(ab); арифметическая даёт 2,5 и 10 — мутация ловится
    assert not np.allclose(M.geo(a, b, 0.5), 0.5 * (a + b))
    assert np.allclose(M.geo(a, b, 0.0), a) and np.allclose(M.geo(a, b, 1.0), b)


def test_known_answer_weight_goes_to_exact_expert_and_sigma_positive():
    rv, har, ml = synth()
    d, td = dates(len(rv))
    s2, w = M.geo_mix(rv, har, ml, d, td)
    assert (s2 > 0).all() and np.isfinite(s2).all()
    assert (w[:252] == 0.5).all() and w[600:].mean() > 0.85                  # до 252 закрытых — w = 0,5; затем вес уходит к точному эксперту


def test_hedge_weights_sum_to_one_and_favor_better_expert():
    rv, har, ml = synth()
    d, td = dates(len(rv))
    s2, W, eta = M.hedge_mix(rv, np.column_stack([har, ml]), d, td)
    assert np.allclose(W.sum(axis=1), 1.0) and (W[:252] == 0.5).all() and W[600:, 1].mean() > 0.55 and (s2 > 0).all()


def test_truncation_and_closed_plus_one_mutation():
    rv, har, ml = synth()
    d, td = dates(len(rv))
    k = 500
    m = np.ones(len(rv))
    m[k + 1:] = np.random.default_rng(2).lognormal(0, 1, len(rv) - k - 1)
    a, _ = M.geo_mix(rv, har, ml, d, td)
    b, _ = M.geo_mix(rv * m, har * m, ml * m, d, td)
    assert np.array_equal(a[:k + 1], b[:k + 1])
    m2, har2 = np.ones(len(rv)), har.copy()
    m2[k:] = 1e8                                                             # портим только цель с дня k (RV d_k+1 — будущее на d_k)
    har2[k] *= 1e4                                                           # на дне k HAR-эксперт далеко выше: с утёкшей целью выбор w обязан сместиться к нему
    ok1, ok2 = M.geo_mix(rv, har2, ml, d, td)[0], M.geo_mix(rv * m2, har2, ml, d, td)[0]
    bad1, bad2 = M.geo_mix(rv, har2, ml, d, td, extra=1)[0], M.geo_mix(rv * m2, har2, ml, d, td, extra=1)[0]
    assert np.array_equal(ok1[:k + 1], ok2[:k + 1]) and not np.array_equal(bad1[:k + 1], bad2[:k + 1])
    h1, h2 = M.hedge_mix(rv, np.column_stack([har, ml]), d, td)[0], M.hedge_mix(rv * m, np.column_stack([har * m, ml * m]), d, td)[0]
    assert np.array_equal(h1[:k + 1], h2[:k + 1])
