"""V4-B: решатели. B2 — проекционный метод Ньютона на ящике (выпуклая функция, аналитические градиент и гессиан); B6 — SLSQP на симплексах (a+b+c = 1) с аналитическим градиентом.
Целевая функция везде — средний (по строкам обучения) QLIKE + штрафы иерархии: (c/2)·Σ_i‖θ_i − θ_panel‖² + (λ/2)·‖θ_panel − θ₀‖², c = κ·s², λ = сила приора·s², s² — средняя кривизна QLIKE
на закрытых парах (mean ln²(M/H)): «κ пар-эквивалентов» = κ псевдо-пар с кривизной настоящей пары."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def newton_box(fgh, x0, lo, hi, tol: float = 1e-10, max_it: int = 80):
    """Проекционный Ньютон с активным набором и Армихо. fgh(x) -> (f, g, H). Возвращает (x, f, сошлось)."""
    x = np.clip(x0, lo, hi)
    f, g, Hm = fgh(x)
    for _ in range(max_it):
        act = ((x <= lo + 1e-14) & (g > 0)) | ((x >= hi - 1e-14) & (g < 0))
        free = ~act
        pg = np.where(act, 0.0, g)
        if np.abs(pg).max() < tol:
            return x, f, True
        d = np.zeros_like(x)
        d[free] = np.linalg.solve(Hm[np.ix_(free, free)], -g[free])
        t = 1.0
        while True:
            xn = np.clip(x + t * d, lo, hi)
            fn, gn, Hn = fgh(xn)
            if fn <= f + 1e-4 * float(g @ (xn - x)) or t < 1e-12:
                break
            t *= 0.5
        x, f, g, Hm = xn, fn, gn, Hn
    return x, f, bool(np.abs(np.where(((x <= lo + 1e-14) & (g > 0)) | ((x >= hi - 1e-14) & (g < 0)), 0.0, g)).max() < 1e-7)


def b2_fgh(T, s2, kappa, lam, w0):
    """Целевая функция B2 по x = [w_panel, w_1..w_n] -> (f, g, H) (средняя по строкам)."""
    n, N, c, l = T.n, len(T.rv), kappa * s2, lam * s2
    r = T.lH - T.lM

    def fgh(x):
        wp, w = x[0], x[1:]
        lF = w[T.tid] * T.lH + (1 - w[T.tid]) * T.lM
        e = T.rv / np.exp(lF)
        dev = w - wp
        f = ((e + lF).sum() + 0.5 * c * float(dev @ dev) + 0.5 * l * (wp - w0) ** 2) / N
        g = np.empty(n + 1)
        g[1:] = (np.bincount(T.tid, (1 - e) * r, minlength=n) + c * dev) / N
        g[0] = (-c * dev.sum() + l * (wp - w0)) / N
        H = np.zeros((n + 1, n + 1))
        H[np.arange(1, n + 1), np.arange(1, n + 1)] = (np.bincount(T.tid, e * r * r, minlength=n) + c) / N
        H[0, 1:] = H[1:, 0] = -c / N
        H[0, 0] = (n * c + l) / N
        return f, g, H
    return fgh


def b6_obj(T, s2, kappa, lam, th0):
    """Целевая функция B6 по x = (n+1, 3), строка 0 — панель -> (f, g)."""
    n, N, c, l = T.n, len(T.rv), kappa * s2, lam * s2
    L = np.column_stack([T.lH, T.lK2, T.lK3])

    def obj(x):
        X = x.reshape(n + 1, 3)
        th = X[1 + T.tid]
        lF = (th * L).sum(1)
        e = T.rv / np.exp(lF)
        dev = X[1:] - X[0]
        f = ((e + lF).sum() + 0.5 * c * float((dev ** 2).sum()) + 0.5 * l * float(((X[0] - th0) ** 2).sum())) / N
        G = np.zeros_like(X)
        for j in range(3):
            G[1:, j] = np.bincount(T.tid, (1 - e) * L[:, j], minlength=n)
        G[1:] += c * dev
        G[0] = -c * dev.sum(0) + l * (X[0] - th0)
        return f, (G / N).ravel()
    return obj


def solve_b6(T, s2, kappa, lam, th0, x0):
    """SLSQP: каждая строка (панель и бумаги) — на симплексе. -> (X (n+1, 3), f, сошлось)."""
    n = T.n
    A = np.zeros((n + 1, 3 * (n + 1)))
    for i in range(n + 1):
        A[i, 3 * i:3 * i + 3] = 1.0
    cons = [{"type": "eq", "fun": lambda x: A @ x - 1.0, "jac": lambda x: A}]
    r = minimize(b6_obj(T, s2, kappa, lam, th0), np.asarray(x0, float).ravel(), jac=True, method="SLSQP", bounds=[(0.0, 1.0)] * (3 * (n + 1)), constraints=cons,
                 options={"maxiter": 300, "ftol": 1e-14})
    return r.x.reshape(n + 1, 3), float(r.fun), bool(r.success)
