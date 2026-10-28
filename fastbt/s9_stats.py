"""С9: статистика для доводки стратегий — разность Шарпа по Ледуа–Вольфу (2008), плата
Флеминга–Кирби–Остдика (2001), признаки затворов (ранг по прошлому окну, гистерезис, полоса).

Все окна — только прошлые бары (сдвиг на 1): значение бара t сравнивается с окном [t−W, t−1].
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numba import njit


# ---------- признаки по прошлому окну ----------

@njit(cache=True)
def past_rank(s, W, minp):
    """rank_t = доля значений s[t−W..t−1] (без NaN) строго меньше s_t; NaN при < minp значений
    окна или NaN в s_t. Текущий бар в окно не входит."""
    n = len(s)
    out = np.full(n, np.nan)
    for t in range(n):
        v = s[t]
        if v != v:
            continue
        lo = t - W if t - W > 0 else 0
        c = 0
        k = 0
        for j in range(lo, t):
            x = s[j]
            if x == x:
                c += 1
                if x < v:
                    k += 1
        if c >= minp:
            out[t] = k / c
    return out


def minp(W: int) -> int:
    return max(50, W // 5)


def past_median(s: np.ndarray, W: int) -> np.ndarray:
    """Медиана s[t−W..t−1] (сдвиг на 1), min max(50, W/5) значений."""
    return pd.Series(s).shift(1).rolling(W, min_periods=minp(W)).median().values


def ema(s: np.ndarray, span: int) -> np.ndarray:
    """EMA (только прошлое и текущее значение), NaN сохраняются."""
    return pd.Series(s).ewm(span=int(span), min_periods=1, ignore_na=True).mean().where(
        ~np.isnan(s)).values


@njit(cache=True)
def band_hold(w, band):
    """Полоса бездействия: держим вес, пока |w_t / w_held − 1| ≤ band; NaN → держим (старт 1)."""
    n = len(w)
    out = np.empty(n)
    cur = 1.0
    for t in range(n):
        v = w[t]
        if v == v and v > 0:
            if band <= 0 or abs(v / cur - 1.0) > band:
                cur = v
        out[t] = cur
    return out


# ---------- Ледуа–Вольф (2008): разность Шарпа ----------

def _lw_parts(x1, x2):
    m1, m2 = x1.mean(), x2.mean()
    g1, g2 = (x1 ** 2).mean(), (x2 ** 2).mean()
    d = m1 / math.sqrt(g1 - m1 ** 2) - m2 / math.sqrt(g2 - m2 ** 2)
    grad = np.array([g1 / (g1 - m1 ** 2) ** 1.5, -g2 / (g2 - m2 ** 2) ** 1.5,
                     -0.5 * m1 / (g1 - m1 ** 2) ** 1.5, 0.5 * m2 / (g2 - m2 ** 2) ** 1.5])
    y = np.column_stack([x1 - m1, x2 - m2, x1 ** 2 - g1, x2 ** 2 - g2])
    return d, grad, y


def _hac(y):
    """HAC (Ньюи–Уэст, окно Бартлетта, L = floor(4·(T/100)^(2/9)))."""
    T = len(y)
    L = int(4 * (T / 100.0) ** (2.0 / 9.0))
    S = y.T @ y / T
    for k in range(1, L + 1):
        G = y[k:].T @ y[:-k] / T
        S += (1 - k / (L + 1)) * (G + G.T)
    return S


def lw_sharpe_diff(x1, x2, reps=1000, block=5, seed=42):
    """Δ = SR(x1) − SR(x2) (годовые), HAC-с.о., p — студентизированный круговой блочный
    бутстрап (Ledoit & Wolf 2008, раздел 3.2): Ψ* = (1/(l·b)) Σ ζ_j ζ_jᵀ по блокам ресэмпла."""
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    T = len(x1)
    d, grad, y = _lw_parts(x1, x2)
    se = math.sqrt(max(grad @ _hac(y) @ grad, 1e-30) / T)
    l = T // block
    Tb = l * block
    rng = np.random.RandomState(seed)
    xx1 = np.concatenate([x1, x1[:block]])
    xx2 = np.concatenate([x2, x2[:block]])
    stat = abs(d) / se
    cnt = 0
    for _ in range(reps):
        st = rng.randint(0, T, l)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()
        b1, b2 = xx1[idx], xx2[idx]
        ds, gs, ys = _lw_parts(b1, b2)
        z = ys.reshape(l, block, 4).sum(axis=1)
        psi = z.T @ z / Tb
        ses = math.sqrt(max(gs @ psi @ gs, 1e-30) / Tb)
        if abs(ds - d) / ses >= stat:
            cnt += 1
    p = (cnt + 1) / (reps + 1)
    return d * math.sqrt(252), se * math.sqrt(252), p


# ---------- Флеминг–Кирби–Остдик (2001): плата за переход ----------

def fko_fee(r1, r0, gamma):
    """Δ (% годовых): средняя квадратичная полезность (1+r1−Δ) равна полезности (1+r0);
    U(R) = R − γ/(2(1+γ))·R². r — дневные полные доходности (total). Δ > 0 — инвестор готов
    платить за переход 0 → 1."""
    a = gamma / (2.0 * (1.0 + gamma))
    R1, R0 = 1.0 + np.asarray(r1, float), 1.0 + np.asarray(r0, float)
    u0 = np.mean(R0 - a * R0 ** 2)
    m1, q1 = R1.mean(), (R1 ** 2).mean()
    # −aΔ² + Δ(2a·m1 − 1) + (m1 − a·q1 − u0) = 0 → корень около нуля
    A, B, C = -a, 2 * a * m1 - 1.0, m1 - a * q1 - u0
    disc = B * B - 4 * A * C
    roots = [(-B + s * math.sqrt(max(disc, 0.0))) / (2 * A) for s in (1, -1)]
    dlt = min(roots, key=abs)
    return float(dlt * 252 * 100)
