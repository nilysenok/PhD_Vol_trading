"""Z2: синтетика для тестов (известный ответ). Склад: дни, r (n, 53) со слабой тяжёлой хвостовой частью и редкими пропусками; Hawkes-события (дискретная постановка плана) с заданными (μ, n, β, η, ρ) и меткой m ~ Exp(3); контекст без порогов."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from models_v2.s16 import x1_z2_hawkes as HK
from models_v2.s16.x1_z2_ev import DELTA, NS


def store(n: int = 300, seed: int = 0, absent: float = 0.01) -> tuple:
    rng = np.random.default_rng(seed)
    r = rng.standard_t(4, size=(n, NS)) * 0.1
    r[rng.random((n, NS)) < absent] = np.nan
    return pd.bdate_range("2016-01-04", periods=n), r


def ctx_from_events(days, N, m, pres) -> dict:
    """Контекст ступени 1 по готовым событиям (порог не нужен: все дни в правдоподобии)."""
    gaps = np.diff(days.values).astype("timedelta64[D]").astype(int) - 1
    return dict(days=days, pres=pres, N=N.astype(float), m=m, r2=np.where(pres, 1.0, 0.0), dg=np.r_[0.0, gaps], dnext=np.r_[gaps, 0].astype(float), base_inlik=np.ones(len(days), bool))


def simulate(mu: float, n: float, beta: float, eta: float, rho: float, n_days: int, seed: int = 0, f=None) -> dict:
    """Дискретный Hawkes плана: P(N = 1) = 1 − exp(−δλ), λ = μ f_k + H, H_{k+1} = e^{−βδ} H_k + α N e^{ηm}; ночь: ρ^{1+Δ}; m ~ Exp(3), M = 3/(3 − η)."""
    rng, f = np.random.default_rng(seed), (np.ones(NS) if f is None else f)
    days = pd.bdate_range("2016-01-04", periods=n_days)
    gaps = np.r_[0, np.diff(days.values).astype("timedelta64[D]").astype(int) - 1]
    E = math.exp(-beta * DELTA)
    alpha = n * (1.0 - E) / (DELTA * 3.0 / (3.0 - eta))
    N, m, H = np.zeros((n_days, NS)), np.zeros((n_days, NS)), 0.0
    for j in range(n_days):
        H *= rho ** (1.0 + gaps[j]) if j else 1.0
        for k in range(NS):
            if rng.random() < 1.0 - math.exp(-DELTA * (mu * f[k] + H)):
                N[j, k], m[j, k] = 1.0, rng.exponential(1.0 / 3.0)
            H = E * H + alpha * N[j, k] * math.exp(eta * m[j, k])
    return dict(ctx=ctx_from_events(days, N, m, np.ones((n_days, NS), bool)), f=f, N=N, m=m)


def x1_frame(ctx: dict, seed: int = 0) -> tuple:
    """Синтетический ряд пар X1 на днях склада (кроме последнего): d, t_plus (следующая сессия), rv, F = X1; e_d — день с событием."""
    rng, days = np.random.default_rng(seed), ctx["days"]
    X1 = np.exp(rng.normal(0.0, 0.3, len(days) - 1))
    rv = X1 * np.exp(rng.normal(0.0, 0.5, len(days) - 1) + 0.3 * (ctx["N"][:-1].sum(1) > 0))
    x1 = pd.DataFrame(dict(d=days[:-1], t_plus=days[1:], rv=rv, F=X1))
    return x1, (ctx["N"][:-1].sum(1) > 0).astype(int)


def window_of(ctx: dict, warm: int = HK.WARM) -> dict:
    """Окно на весь синтетический ряд (первые `warm` строк — прогрев)."""
    return HK.rows(ctx, warm, len(ctx["N"]), warm)
