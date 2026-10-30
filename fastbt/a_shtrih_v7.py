"""Л3.1 A'-расширенная: веса Sk — паритет риска с усадкой Ledoit-Wolf (Ж-093 п. 6, Ж-146 п. 3).

Только numpy. Окно 252 строк строго < i (решение дня i видит данные <= i-1, Д2).
Контроль — равный риск w ~ 1/sigma. Max-Sharpe и любая оптимизация по доходности запрещены.
"""

from __future__ import annotations

import numpy as np

REZHIMY = ("paritet", "ravnyj_risk")


def _proverka(R: np.ndarray, min_t: int = 2) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    if R.ndim != 2 or R.shape[0] < min_t or R.shape[1] < 1:
        raise ValueError(f"нужна матрица [T>={min_t}, S>=1], получено {R.shape}")
    if not np.all(np.isfinite(R)):
        raise ValueError("в матрице доходностей есть NaN/inf")
    return R


def _moments(R: np.ndarray):
    T = R.shape[0]
    x = R - R.mean(axis=0)
    S = x.T @ x / T
    x2 = x * x
    pi = x2.T @ x2 / T - S * S  # pi_ij = mean_t (x_ti x_tj - s_ij)^2
    return S, pi, T


def lw_delta(R: np.ndarray) -> float:
    """Интенсивность усадки Ledoit-Wolf (2004) к цели F = diag(S).

    kappa = (pi - rho) / gamma; pi = sum_ij pi_ij, rho = 0 вне диагонали
    (f_ij = 0 — константа), gamma = sum_{i!=j} s_ij^2; delta = clip(kappa / T, 0, 1).
    """
    S, pi, T = _moments(_proverka(R))
    off = ~np.eye(S.shape[0], dtype=bool)
    gam = float((S[off] ** 2).sum())
    if gam == 0.0:
        return 1.0
    return float(min(1.0, max(0.0, pi[off].sum() / gam / T)))


def lw(R: np.ndarray) -> np.ndarray:
    """Оценка ковариации: delta * diag(S) + (1 - delta) * S (корреляции — к нулю)."""
    S, _, _ = _moments(_proverka(R))
    d = lw_delta(R)
    return d * np.diag(np.diag(S)) + (1.0 - d) * S


def ravnyj_risk(sigma: np.ndarray) -> np.ndarray:
    """Контроль: w ~ 1/sigma, сумма 1."""
    s = np.asarray(sigma, dtype=float)
    if s.ndim != 1 or s.size == 0 or not np.all(np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("sigma должна быть конечной и > 0")
    w = 1.0 / s
    return w / w.sum()


def paritet(S: np.ndarray, tol: float = 1e-12, maxit: int = 200) -> np.ndarray:
    """Веса с равным вкладом w_i (S w)_i в риск; сумма 1, w > 0.

    Минимум выпуклой f(y) = y'Sy/2 - (1/n) sum log y_i (метод Ньютона с откатом).
    """
    S = np.asarray(S, dtype=float)
    n = S.shape[0] if S.ndim == 2 else 0
    if n == 0 or S.shape != (n, n) or not np.all(np.isfinite(S)):
        raise ValueError("нужна конечная квадратная матрица")
    if np.any(np.diag(S) <= 0):
        raise ValueError("нулевая или отрицательная дисперсия на диагонали")

    def f(y):
        return 0.5 * y @ S @ y - np.log(y).sum() / n

    y = 1.0 / np.sqrt(np.diag(S))
    for _ in range(maxit):
        g = S @ y - 1.0 / (n * y)
        if np.max(np.abs(y * (S @ y) / (y @ S @ y / n) - 1.0)) < tol:
            break
        step = np.linalg.solve(S + np.diag(1.0 / (n * y * y)), g)
        t = 1.0
        while t > 1e-12 and (np.any(y - t * step <= 0) or f(y - t * step) > f(y)):
            t *= 0.5
        y = y - t * step
    w = y / y.sum()
    rc = w * (S @ w)
    if np.max(np.abs(rc - rc.mean())) > 1e-8 * rc.mean():
        raise ValueError("паритет не сошёлся (матрица не положительно определена?)")
    return w


def vesa_dnya(ex_sk: np.ndarray, i: int, okno: int = 252, rezhim: str = "paritet"):
    """Веса Sk для решения дня i по ex_sk[i-okno:i] (строка i и позже не читаются).

    Возвращает (w[S], флаг): "ok" | "korotkoe_okno" (строк < okno: равный риск по
    имеющимся) | "isklyucheny" (столбец с NaN или нулевой sigma получил вес 0)
    | "net_validnyh" (все веса 0). NaN в весах не бывает.
    """
    ex = np.asarray(ex_sk, dtype=float)
    if ex.ndim != 2 or ex.shape[1] < 1:
        raise ValueError("ex_sk должна быть матрицей [T, S]")
    if rezhim not in REZHIMY:
        raise ValueError(f"режим {rezhim!r} не из {REZHIMY}")
    if not 0 <= i <= ex.shape[0]:
        raise ValueError(f"день i={i} вне ленты длины {ex.shape[0]}")
    n = ex.shape[1]
    win = ex[max(0, i - okno) : i]
    korotko = win.shape[0] < okno
    if win.shape[0] < 2:
        return np.full(n, 1.0 / n), "korotkoe_okno"
    sd = win.std(axis=0)
    ok = np.isfinite(win).all(axis=0) & (np.ptp(np.nan_to_num(win), axis=0) > 0) & (sd > 0)
    w = np.zeros(n)
    if not ok.any():
        return w, "net_validnyh"
    sub = win[:, ok]
    if korotko or rezhim == "ravnyj_risk":
        w[ok] = ravnyj_risk(sub.std(axis=0))
    else:
        w[ok] = paritet(lw(sub))
    return w, "korotkoe_okno" if korotko else ("ok" if ok.all() else "isklyucheny")
