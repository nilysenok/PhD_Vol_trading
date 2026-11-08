"""V4-C, G16: гейт условной смеси — тождество с B2, известный ответ на синтетике (знак и порядок величины), градиент, приор, пустой набор признаков."""
import numpy as np
import pytest
from scipy.optimize import check_grad

from models_v2.s11 import v4_b_fit as FT, v4_b_ops as OP, v4_c_fit as CF


def test_identity_with_b2_forward_map():
    """β = 0, δ = 0, ℓ-константа: F = H·exp(σ(α)·ℓ) = B2 с w_H = 1 − σ(α) (допуск 1e-8)."""
    rng = np.random.default_rng(1)
    H, K2 = np.exp(rng.normal(0, 0.5, 500)), np.exp(rng.normal(0, 0.5, 500))
    M = OP.mean_m(K2, K2 * 1.1)
    wh = rng.uniform(0.05, 0.95, 500)
    F, _ = CF.forward(np.log(H), np.log(M / H), CF.logit_w_of_b2(wh))
    assert np.allclose(F, OP.forward("B2", H, M, wh)[0], rtol=1e-8, atol=0)


def test_mutation_weight_on_wrong_side_is_caught():
    """Мутация «вес на H вместо M» (F = M·exp(w·ln(H/M))) расходится с B2 при w_H = 1 − σ(α)."""
    H, M, wh = np.array([1.0, 2.0]), np.array([2.0, 1.0]), np.array([0.3, 0.7])
    F, w = CF.forward(np.log(H), np.log(M / H), CF.logit_w_of_b2(wh))
    wrong = np.exp(np.log(M) + w * np.log(H / M))
    assert not np.allclose(wrong, OP.forward("B2", H, M, wh)[0], rtol=1e-8)


def test_unpenalized_optimum_equals_b2_optimum():
    """Без штрафов и признаков оптимум C (n = 1) даёт тот же прогноз, что оптимум B2: те же данные, тот же QLIKE."""
    rng = np.random.default_rng(2)
    N = 4000
    H = np.exp(rng.normal(0, 0.4, N))
    M = H * np.exp(rng.normal(0, 0.3, N))
    rv = np.sqrt(H * M) * rng.exponential(1.0, N)
    r = CF.fit(rv, np.log(H), np.log(M / H), np.zeros((N, 0)), np.zeros(N, int), 1, kappa=1e-9, lam=1e-9)     # штраф исчезающе мал (при нуле w_панели в B2 не определён)
    T, _ = FT.make_train(rv, H, M, M, np.zeros(N, int))
    b = FT.fit("B2", T, kappa=1e-9, lam=1e-9)
    assert r["ok"] and b["ok"]
    wc = CF.sigmoid(r["alpha"] + r["delta"][0])
    assert wc == pytest.approx(1.0 - b["w"][0], abs=1e-5)


def test_known_gate_sign_and_order_recovered():
    """w = σ(−0,3 + 1,2·z): оценка β положительна, порядок величины сохранён; α близка к −0,3; признак-шум получает β ≈ 0."""
    rng = np.random.default_rng(3)
    N = 30000
    z, noise = rng.normal(0, 1, N), rng.normal(0, 1, N)
    H = np.exp(rng.normal(0, 0.3, N))
    ell = rng.normal(0, 0.6, N)
    w = CF.sigmoid(-0.3 + 1.2 * z)
    rv = H * np.exp(w * ell) * rng.gamma(25.0, 1.0 / 25.0, N)       # E[rv/F] = 1, шум умеренный: оценка β сходится к 1,2 с ошибкой < 0,3
    Z = np.column_stack([z, noise])
    r0 = CF.fit(rv, np.log(H), ell, Z, np.zeros(N, int), 1, kappa=1e-6, lam=1e-6)       # штрафа практически нет: восстановление точное
    assert r0["ok"] and abs(r0["beta"][0] - 1.2) < 0.1 and abs(r0["alpha"] + r0["delta"][0] + 0.3) < 0.1 and abs(r0["beta"][1]) < 0.05
    r = CF.fit(rv, np.log(H), ell, Z, np.zeros(N, int), 1)                              # объявленный штраф (κ = λ = 250 пар-эквивалентов × s²): знак и порядок сохранены, β сжата к 0
    assert r["ok"] and 0.5 < r["beta"][0] < 1.2 and r["beta"][0] < r0["beta"][0] and abs(r["beta"][1]) < 0.1


def test_known_gate_mutation_z_sign_flipped_is_caught():
    """Подложенный знак признака (z → −z) даёт β < 0 — тест знака ловит."""
    rng = np.random.default_rng(3)
    N = 30000
    z = rng.normal(0, 1, N)
    H = np.exp(rng.normal(0, 0.3, N))
    ell = rng.normal(0, 0.6, N)
    rv = H * np.exp(CF.sigmoid(1.2 * z) * ell) * rng.exponential(1.0, N)
    assert CF.fit(rv, np.log(H), ell, (-z)[:, None], np.zeros(N, int), 1)["beta"][0] < -0.8


def test_analytic_gradient_matches_numeric():
    rng = np.random.default_rng(4)
    N, k, n = 300, 3, 4
    tid = rng.integers(0, n, N)
    Z, H, ell = rng.normal(size=(N, k)), np.exp(rng.normal(0, 0.3, N)), rng.normal(0, 0.5, N)
    rv = H * rng.exponential(1.0, N)
    f = CF._obj(rv, np.log(H), ell, Z, tid, n, 5.0, 3.0)
    x0 = rng.normal(0, 0.3, n + 1 + k)
    assert check_grad(lambda x: f(x)[0], lambda x: f(x)[1], x0) < 1e-6


def test_penalty_shrinks_beta_and_delta():
    """Ридж к 0: с большим штрафом β и δ уменьшаются по модулю (гиперпараметры κ, λ фиксированы, здесь только направление)."""
    rng = np.random.default_rng(5)
    N = 2000
    z = rng.normal(size=(N, 1))
    H, ell = np.exp(rng.normal(0, 0.3, N)), rng.normal(0, 0.6, N)
    rv = H * np.exp(CF.sigmoid(1.0 * z[:, 0]) * ell) * rng.exponential(1.0, N)
    tid = rng.integers(0, 3, N)
    lo, hi = CF.fit(rv, np.log(H), ell, z, tid, 3, kappa=1e-3, lam=1e-3), CF.fit(rv, np.log(H), ell, z, tid, 3, kappa=1e5, lam=1e5)
    assert abs(hi["beta"][0]) < abs(lo["beta"][0]) and np.abs(hi["delta"]).max() < np.abs(lo["delta"]).max() + 1e-12


def test_standardize_constant_feature_sd_one():
    mu, sd = CF.standardize(np.column_stack([np.ones(10), np.arange(10.0)]))
    assert sd[0] == 1.0 and sd[1] == pytest.approx(np.arange(10.0).std())
