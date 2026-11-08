"""V5 лист 1a (Ж-226), Graph-Score: ядро фильтра состояния u (numba) с аналитическими чувствительностями к θ = (ρ, α, β).
F(i,τ) = B(i,τ)·exp(u(i,τ)), B — замороженная основа (K8; при g₀ = 0 — H); s = RV/F − 1; s̃ = s или min(s, 3).
u(i, следующая строка) = ρ·u + α·s̃(i,τ) + β·T(i,τ), u в первой строке = 0. T(i,τ) = c_i·Σ_j W[i,j]·s̃(j,τ)·p_j / Σ_j W[i,j]·p_j — по скорам других бумаг ТОГО ЖЕ дня τ (p_j: бумага j в день τ торговалась и её скор питает остальных).
Ось — даты цели τ (t⁺); строка (бумага, τ) относится к прогнозу, сделанному в день d < τ: все скоры, входящие в u строки, имеют дату ≤ d (проверяет `panel.check_closed`)."""
from __future__ import annotations

import numba as nb
import numpy as np

CLIP = 3.0


@nb.njit(cache=False)
def _step_scores(n, rv, base, pres, lm, u, du, clip, x, dx, S, X, U, acc):
    m = rv.shape[1]
    for i in range(m):
        if not pres[n, i]:
            continue
        r = rv[n, i] / (base[n, i] * np.exp(u[i]))
        s = r - 1.0
        U[n, i], S[n, i] = u[i], s
        if clip and s > CLIP:
            x[i] = CLIP
            dx[i, :] = 0.0
        else:
            x[i] = s
            dx[i, :] = -r * du[i, :]
        X[n, i] = x[i]
        if lm[i]:
            acc[0] += r - np.log(r) - 1.0
            acc[1] += 1.0
            acc[2:5] -= s * du[i, :]                  # dQLIKE/dθ = −s·du
            acc[5:8] += du[i, :] ** 2


@nb.njit(cache=False)
def _step_update(n, pres, fm, W, c, th, u, du, x, dx):
    m = len(u)
    new_u, new_du = u.copy(), du.copy()
    for i in range(m):
        if not pres[n, i]:
            continue
        num, den = 0.0, 0.0
        dnum = np.zeros(3)
        for j in range(m):
            if j != i and pres[n, j] and fm[j]:
                num += W[i, j] * x[j]
                den += W[i, j]
                dnum += W[i, j] * dx[j, :]
        T, dT = 0.0, np.zeros(3)
        if den > 0.0:
            T, dT = c[i] * num / den, c[i] * dnum / den
        new_u[i] = th[0] * u[i] + th[1] * x[i] + th[2] * T
        new_du[i, :] = th[0] * du[i, :] + th[1] * dx[i, :] + th[2] * dT
        new_du[i, 0] += u[i]
        new_du[i, 1] += x[i]
        new_du[i, 2] += T
    return new_u, new_du


@nb.njit(cache=False)
def filt(rv, base, pres, lm, fm, W, c, th, clip, n_use):
    """-> (U, S, X, acc). U[n,i] — u в F строки; S — сырой скор; X — используемый s̃; acc = [Σ QLIKE, число строк, ∇θ Σ QLIKE (3), Σ du² (3)] по строкам с lm[i] и n < n_use."""
    N, m = rv.shape
    u, du = np.zeros(m), np.zeros((m, 3))
    U, S, X = np.zeros((N, m)), np.zeros((N, m)), np.zeros((N, m))
    x, dx, acc = np.zeros(m), np.zeros((m, 3)), np.zeros(8)
    for n in range(min(n_use, N)):
        _step_scores(n, rv, base, pres, lm, u, du, clip, x, dx, S, X, U, acc)
        u, du = _step_update(n, pres, fm, W, c, th, u, du, x, dx)
    return U, S, X, acc


def equal_weights(m: int) -> tuple:
    """Матрица W (равные веса j ≠ i) и c = 1: передача T = среднее s̃ остальных бумаг дня (GS3); GS1/GS2 используют ту же матрицу при β = 0."""
    return np.ones((m, m)) - np.eye(m), np.ones(m)


def run(rv, base, pres, th, W=None, c=None, clip=False, lm=None, fm=None, n_use=None) -> tuple:
    """Обёртка `filt` с умолчаниями (все бумаги питают друг друга и входят в потери)."""
    N, m = rv.shape
    W0, c0 = equal_weights(m)
    W, c = (W0 if W is None else W), (c0 if c is None else c)
    lm = np.ones(m, bool) if lm is None else lm
    fm = np.ones(m, bool) if fm is None else fm
    return filt(np.ascontiguousarray(rv, float), np.ascontiguousarray(base, float), pres, lm, fm, np.ascontiguousarray(W, float), np.ascontiguousarray(c, float),
                np.asarray(th, float), bool(clip), N if n_use is None else int(n_use))
