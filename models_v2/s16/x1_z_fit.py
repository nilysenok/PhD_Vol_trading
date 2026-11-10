"""Этап Z, обучение (план Z раздел 3–4). Потеря — QLIKE прогноза F = X1·e^g: l = u·e^{−g} + g − ln u − 1, u = RV(t⁺)/X1; dl/dg = 1 − u·e^{−g}. Цель = mean l + λ·(штраф); L-BFGS-B (scipy), ≤ 500 итераций, детерминированный старт без seed.
Линейная семья (ZA, ZB, ZB0): g = b + a·s (на ZB, ZB0 — строки e_d = 1; ZB0: a ≡ 0 границами); штраф λ(b² + a²); |a|, |b| ≤ 5.
Затвор (ZC, ZCp): g = σ(η)·h, η = γ0 + γᵀz̃, h = b + a·s; штраф λ(|γ|² + b² + a²) (γ0 не штрафуется); |γ0|, |γ|, ≤ 10 (трактовка: границу γ0 плана не задаёт — взята как у γ); старт γ0 = −2, γ = 0, b = a = 0.
Выбор λ (и ω у ZA): обучение на закрытых парах без последних V, потеря на последних V; V = 126 при n ≥ 252, иначе ⌊n/2⌋ (ZB, ZB0 — V_e = 60 при n_e ≥ 120 по ЭКСТРЕМАЛЬНЫМ парам); ничья — больший λ, затем меньший ω; затем переобучение на всём пуле. Чистые функции."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

LAMS, OMEGAS, MAXITER, BOUND_BA, BOUND_G = (0.0, 0.001, 0.01, 0.1), (0.0, 0.25, 0.5, 0.75, 1.0), 500, 5.0, 10.0
V_FULL, V_EXT, MIN_EXT, START_G0 = 126, 60, 30, -2.0


def qlike_g(u, g) -> float:
    return float(np.mean(u * np.exp(-g) + g - np.log(u) - 1.0))


def lin_loss_grad(th, u, s, lam):
    """th = (b, a); g = b + a·s."""
    g = th[0] + th[1] * s
    ug = u * np.exp(-g)
    dg = 1.0 - ug
    return float(np.mean(ug + g - np.log(u) - 1.0)) + lam * float(th @ th), np.array([np.mean(dg) + 2 * lam * th[0], np.mean(dg * s) + 2 * lam * th[1]])


def fit_lin(u, s, lam, free_a: bool = True) -> np.ndarray:
    """-> (b, a); free_a = False (ZB0): a ≡ 0."""
    bnd = [(-BOUND_BA, BOUND_BA), (-BOUND_BA, BOUND_BA) if free_a else (0.0, 0.0)]
    return np.asarray(minimize(lin_loss_grad, np.zeros(2), args=(u, s, lam), jac=True, method="L-BFGS-B", bounds=bnd, options=dict(maxiter=MAXITER)).x, float)


def gate_parts(th, s, Zt):
    p = 1.0 / (1.0 + np.exp(-(th[0] + (Zt * th[1:5]).sum(1))))          # без matmul (ложные предупреждения Accelerate на macOS)
    return p, th[5] + th[6] * s


def gate_loss_grad(th, u, s, Zt, lam):
    """th = (γ0, γ1…γ4, b, a); g = σ(γ0 + γᵀz̃)·(b + a·s). Аналитический градиент."""
    p, h = gate_parts(th, s, Zt)
    g = p * h
    ug = u * np.exp(-g)
    dg = 1.0 - ug
    c = dg * h * p * (1.0 - p)                                    # dl/dη
    gr = np.concatenate([[np.mean(c)], (c[:, None] * Zt).mean(0) + 2 * lam * th[1:5], [np.mean(dg * p) + 2 * lam * th[5], np.mean(dg * p * s) + 2 * lam * th[6]]])
    return float(np.mean(ug + g - np.log(u) - 1.0)) + lam * float(np.sum(th[1:] ** 2)), gr


def gate_start() -> np.ndarray:
    x = np.zeros(7)
    x[0] = START_G0
    return x


def fit_gate(u, s, Zt, lam) -> np.ndarray:
    bnd = [(-BOUND_G, BOUND_G)] * 5 + [(-BOUND_BA, BOUND_BA)] * 2
    return np.asarray(minimize(gate_loss_grad, gate_start(), args=(u, s, Zt, lam), jac=True, method="L-BFGS-B", bounds=bnd, options=dict(maxiter=MAXITER)).x, float)


def gate_g(th, s, Zt) -> np.ndarray:
    p, h = gate_parts(th, s, Zt)
    return p * h


def n_valid(n: int, full: int) -> int:
    return full if n >= 2 * full else n // 2


def val_idx(pool: np.ndarray, v: int) -> np.ndarray:
    """Последние V ЗАКРЫТЫХ пар пула (мутация: V пар ПОСЛЕ последней закрытой — будущее)."""
    return pool[-v:]


def ext_pool(pool: np.ndarray, e) -> np.ndarray:
    """Закрытые экстремальные пары (e_d = 1) пула."""
    return pool[np.asarray(e)[pool] == 1]


def val_block_b(pool: np.ndarray, ext: np.ndarray, v: int) -> np.ndarray:
    """ZB, ZB0: блок валидации — последние V_e экстремальные пары (мутация: последние V пар пула — все дни)."""
    return val_idx(ext, v)


def select(tr, val, fit_fn, score_fn, omegas=(1.0,)) -> tuple:
    """-> (λ, ω): минимум потери на val; ничья — больший λ (перебор по убыванию), затем меньший ω (по возрастанию; строгое «<» сохраняет первое)."""
    best, arg = np.inf, (LAMS[-1], omegas[0])
    for lam in sorted(LAMS, reverse=True):
        th = fit_fn(tr, lam)
        for om in sorted(omegas):
            q = score_fn(th, val, om)
            if q < best:
                best, arg = q, (float(lam), float(om))
    return arg
