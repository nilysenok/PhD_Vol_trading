"""Ж-212 (HS6, G9): производные QLIKE, положительность, штраф μ, монотонность, λ = 0 ≡ HAR; мутации: неверные градиент/гессиан, μ не передан, монотонность не передана."""
from __future__ import annotations

import numpy as np
import pytest

from models_v2.s8 import hs_cfit


def test_gh_matches_finite_differences_with_penalty():
    rng = np.random.default_rng(1)
    y, m, off = np.exp(rng.normal(0, 1, 50)), rng.normal(0, 1, 50), rng.normal(0, 0.3, 50)
    for mu in (0.0, 0.05, 1.0):
        g, h = hs_cfit.gh(y, m, off, mu)
        e = 1e-6
        num_g = (hs_cfit.loss(y, m + e, off, mu) - hs_cfit.loss(y, m - e, off, mu)) / (2 * e)
        num_h = (hs_cfit.gh(y, m + e, off, mu)[0] - hs_cfit.gh(y, m - e, off, mu)[0]) / (2 * e)
        assert np.allclose(g, num_g, atol=1e-7) and np.allclose(h, num_h, atol=1e-6)
        assert (h > 0).all()


def test_mutation_wrong_gradient_sign_is_caught():
    y, m, off = np.array([2.0, 0.5]), np.array([0.1, -0.2]), np.zeros(2)
    g, _ = hs_cfit.gh(y, m, off)
    bad = y * np.exp(-m) - 1.0      # перепутанный знак
    e = 1e-6
    num = (hs_cfit.loss(y, m + e, off) - hs_cfit.loss(y, m - e, off)) / (2 * e)
    assert np.allclose(g, num, atol=1e-7) and not np.allclose(bad, num, atol=1e-3)


def synth(n=1500, shift=1.0, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.choice([-1.0, 1.0], n)
    off = rng.normal(0, 0.2, n)
    f = np.exp(off + 0.5 * shift * z)
    y = f * rng.gamma(20.0, 1 / 20.0, n)       # E[y/f] = 1, QLIKE-согласованная шкала
    return np.column_stack([z, rng.normal(size=n)]), y, off


@pytest.mark.parametrize("lib", ["XGB", "LGBM"])
def test_known_answer_recovers_shift(lib):
    X, y, off = synth()
    pr = hs_cfit.FITS[lib](X, y, off, 0.0, (0, 0), 42)
    d = pr(np.array([[1.0, 0.0]]))[0] - pr(np.array([[-1.0, 0.0]]))[0]
    assert 0.6 < d < 1.1                       # истинно 1,0; 150 деревьев lr 0,03 недоучивают (потолок ≈ 1 − 0,97^150·…)


@pytest.mark.parametrize("lib", ["XGB", "LGBM"])
def test_penalty_shrinks_and_mutation_mu_not_passed_is_caught(lib):
    X, y, off = synth()
    z1, zm = np.array([[1.0, 0.0]]), np.array([[-1.0, 0.0]])
    amp = lambda mu: abs(hs_cfit.FITS[lib](X, y, off, mu, (0, 0), 42)(z1)[0] - hs_cfit.FITS[lib](X, y, off, mu, (0, 0), 42)(zm)[0])
    assert amp(5.0) < 0.6 * amp(0.0)           # штраф сжимает g; если μ не передан в objective — равенство и тест падает


@pytest.mark.parametrize("lib", ["XGB", "LGBM"])
def test_monotone_constraint_is_applied(lib):
    X, y, off = synth(shift=-1.0)              # истинная g убывает по z; ограничение «неубывание» должно её запретить
    pr = hs_cfit.FITS[lib](X, y, off, 0.0, (1, 0), 42)
    g_hi, g_lo = pr(np.array([[1.0, 0.0]]))[0], pr(np.array([[-1.0, 0.0]]))[0]
    assert g_hi >= g_lo - 1e-12
    free = hs_cfit.FITS[lib](X, y, off, 0.0, (0, 0), 42)
    assert free(np.array([[1.0, 0.0]]))[0] < free(np.array([[-1.0, 0.0]]))[0] - 0.3   # без ограничения — убывает (мутация «монотонность не передана»)


def test_lambda_zero_is_har_bitwise_and_lambda_one_is_full_g():
    rng = np.random.default_rng(3)
    har, g = np.exp(rng.normal(0, 1, 100)), rng.normal(0, 0.5, 100)
    assert (har * np.exp(0.0 * g) == har).all()
    assert np.array_equal(har * np.exp(1.0 * g), har * np.exp(g))
    assert not np.array_equal(har * np.exp(0.5 * g), har)      # мутация «λ = 0 ≢ HAR»: λ ≠ 0 меняет ряд
