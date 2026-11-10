"""Этап Z (Ж-246), обучение: аналитический градиент = численному (линейная семья и затвор ZC); из старта плана градиент по γ = 0, по b, a ≠ 0; известный ответ (b, a восстанавливаются, g заложен только на области);
границы |a|, |b| ≤ 5, |γ| ≤ 10; ZB0: a ≡ 0; правило ничьей λ/ω; блок валидации ZB — экстремальные пары (мутация «все пары» ловится)."""
from __future__ import annotations

import numpy as np

import models_v2.s16  # noqa: F401
from models_v2.s16 import x1_z_fit as FT, x1_z_synth as SY


def num_grad(f, th, eps=1e-6):
    out = np.zeros(len(th))
    for i in range(len(th)):
        a, b = th.copy(), th.copy()
        a[i] += eps
        b[i] -= eps
        out[i] = (f(a)[0] - f(b)[0]) / (2 * eps)
    return out


def data(n=400, seed=1):
    x1, D = SY.make(n, seed)
    return D["u"], D["s"], D["Z"]


def test_lin_gradient_analytic_equals_numeric() -> None:
    u, s, _ = data()
    th = np.array([0.3, -0.2])
    an = FT.lin_loss_grad(th, u, s, 0.01)[1]
    assert np.allclose(an, num_grad(lambda t: FT.lin_loss_grad(t, u, s, 0.01), th), atol=1e-7)


def test_gate_gradient_analytic_equals_numeric_and_start_point() -> None:
    u, s, Z = data()
    Zt = (Z - Z.mean(0)) / Z.std(0)
    rng = np.random.default_rng(3)
    th = rng.normal(0, 0.5, 7)
    an = FT.gate_loss_grad(th, u, s, Zt, 0.01)[1]
    assert np.allclose(an, num_grad(lambda t: FT.gate_loss_grad(t, u, s, Zt, 0.01), th), atol=1e-7)
    th0 = FT.gate_start()
    assert th0[0] == -2.0 and not th0[1:].any()
    g0 = FT.gate_loss_grad(th0, u, s, Zt, 0.01)[1]
    assert (g0[:5] == 0).all()                      # по γ0 и γ — ровно ноль (h = b + a·s = 0)
    assert abs(g0[5]) > 1e-4 and abs(g0[6]) > 1e-4  # по b и a — не ноль: оптимизатор уходит со старта
    assert np.allclose(g0, num_grad(lambda t: FT.gate_loss_grad(t, u, s, Zt, 0.01), th0), atol=1e-7)


def test_known_answer_lin_recovers_b_a_without_noise_and_zb0_has_a_zero() -> None:
    x1, D = SY.make(500, 2, noise=False)
    ext = np.flatnonzero(D["e"] == 1)
    th = FT.fit_lin(D["u"][ext], D["s"][ext], 0.0)
    assert np.allclose(th, [SY.TRUE["b0"], SY.TRUE["a0"]], atol=1e-4)
    th0 = FT.fit_lin(D["u"][ext], D["s"][ext], 0.0, free_a=False)
    assert th0[1] == 0.0 and abs(th0[0] - SY.TRUE["b0"]) < 0.2         # сдвиг уровня: a ≡ 0


def test_bounds_hold_on_extreme_data() -> None:
    u = np.full(50, np.exp(30.0))                    # цель недостижима: упираемся в границы
    s = np.ones(50)
    th = FT.fit_lin(u, s, 0.0)
    assert np.all(np.abs(th) <= FT.BOUND_BA + 1e-12)
    Zt = np.random.default_rng(0).normal(size=(50, 4))
    g = FT.fit_gate(u, s, Zt, 0.0)
    assert np.all(np.abs(g[:5]) <= FT.BOUND_G + 1e-12) and np.all(np.abs(g[5:]) <= FT.BOUND_BA + 1e-12)


def test_select_tie_goes_to_larger_lambda_then_smaller_omega() -> None:
    fit = lambda idx, lam: np.zeros(2)
    flat = lambda th, idx, om: 1.0
    assert FT.select(np.arange(5), np.arange(5), fit, flat, FT.OMEGAS) == (FT.LAMS[-1], FT.OMEGAS[0])
    only_lam = lambda th, idx, om: 1.0 - 0.1 * (om == 0.5)          # ω = 0,5 строго лучше при любом λ
    assert FT.select(np.arange(5), np.arange(5), fit, only_lam, FT.OMEGAS) == (FT.LAMS[-1], 0.5)


def test_n_valid_rule_and_validation_blocks() -> None:
    assert FT.n_valid(252, FT.V_FULL) == 126 and FT.n_valid(251, FT.V_FULL) == 125 and FT.n_valid(120, FT.V_EXT) == 60 and FT.n_valid(119, FT.V_EXT) == 59 and FT.n_valid(30, FT.V_EXT) == 15
    pool = np.arange(100)
    e = (np.arange(200) % 4 == 0).astype(int)
    ext = FT.ext_pool(pool, e)
    assert (e[ext] == 1).all() and len(ext) == 25 and np.array_equal(FT.val_block_b(pool, ext, 10), ext[-10:]) and np.array_equal(FT.val_idx(pool, 7), pool[-7:])
