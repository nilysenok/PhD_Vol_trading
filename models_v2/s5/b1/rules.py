"""Правила без обучаемых деревьев: H3 (вес HAR в смеси), R3 (логистика режима), R4 (порог переключения). Всё — только прошлое.

Закрытые прогнозы на дату d: те, у которых target_date <= d (`post.closed_count`). Ничья — меньший w / больший θ.
"""
from __future__ import annotations

import numpy as np

from models_v2 import metrics
from models_v2.s3 import post

W = post.W_GRID
TIE = post.TIE
MIN_EXP, MIN_GROUP, ROLL = 252, 60, 252


def _cum(rv, s2h, s2b):
    """C[j, i] = Σ_{k<=j} QLIKE(rv_k, W_i·s2h_k + (1−W_i)·s2b_k)."""
    mix = W[None, :] * s2h[:, None] + (1 - W[None, :]) * s2b[:, None]
    return np.cumsum(metrics.qlike(rv[:, None], mix), axis=0)


def _argw(tot: np.ndarray) -> float:
    return float(W[int(np.flatnonzero(tot <= tot.min() + TIE)[0])])


def h3_weights(kind: str, rv, s2h, s2b, d, td, grp=None, extra_closed: int = 0) -> np.ndarray:
    """kind: 'exp' (все закрытые), 'roll' (последние 252), 'reg' (по группе режима d; < 60 закрытых группы — w_exp).
    До 252 закрытых — 0,5. extra_closed=1 — мутация (пара d в окне)."""
    n, out = len(rv), np.full(len(rv), 0.5)
    m = np.minimum(post.closed_count(np.asarray(d), np.asarray(td)) + extra_closed, n)
    C = _cum(rv, s2h, s2b)
    w_exp = out.copy()
    for k in range(n):
        if m[k] >= MIN_EXP:
            w_exp[k] = _argw(C[m[k] - 1])
    if kind == "exp":
        return w_exp
    if kind == "roll":
        for k in range(n):
            if m[k] >= MIN_EXP:
                lo = C[m[k] - 1 - ROLL] if m[k] > ROLL else 0.0
                out[k] = _argw(C[m[k] - 1] - lo)
        return out
    return _h3_reg(rv, s2h, s2b, m, grp, C, w_exp)


def _h3_reg(rv, s2h, s2b, m, grp, C, w_exp) -> np.ndarray:
    out = w_exp.copy()
    for g in np.unique(grp[np.isfinite(grp)]):
        mem = (grp == g)
        Cg = np.cumsum(metrics.qlike(rv[:, None], W[None, :] * s2h[:, None] + (1 - W[None, :]) * s2b[:, None]) * mem[:, None], axis=0)
        cnt = np.cumsum(mem)
        for k in np.flatnonzero(mem):
            if m[k] >= MIN_EXP and cnt[m[k] - 1] >= MIN_GROUP:
                out[k] = _argw(Cg[m[k] - 1])
    return out


def mix(w, s2h, s2b) -> np.ndarray:
    return w * s2h + (1 - w) * s2b


def theta_by_year(qw, target_year, s2h, s2alt, rv, cand=(1, 2, 3, 4, 5), include_year: bool = False) -> dict:
    """θ_Y = argmax по прошлому: среднее годовых QLIKE правила(θ) по годам [2017, Y−1] (минимум), ничья — больший θ; 2017: θ = 5.
    include_year=True — мутация (год Y в выборе)."""
    ty, res = np.asarray(target_year), {}
    for Y in range(2017, int(ty.max()) + 1):
        yrs = [y for y in range(2017, Y + (1 if include_year else 0)) if (ty == y).any()]
        if not yrs:
            res[Y] = max(cand)
            continue
        sc = {}
        for th in cand:
            s2 = np.where(np.nan_to_num(qw, nan=-1) >= th, s2alt, s2h)
            q = metrics.qlike(rv, s2)
            sc[th] = float(np.mean([q[ty == y].mean() for y in yrs]))
        lo = min(sc.values())
        res[Y] = max(th for th in cand if sc[th] <= lo + TIE)
    return res


def switch(qw, target_year, theta: dict, s2h, s2alt) -> np.ndarray:
    th = np.array([theta[int(y)] for y in target_year])
    return np.where(np.nan_to_num(qw, nan=-1) >= th, s2alt, s2h)


def fit_logit(Z, yb, lam: float = 1.0, iters: int = 25) -> np.ndarray:
    """Ridge-логистика (IRLS), без штрафа на константе. -> beta (константа первой)."""
    A = np.column_stack([np.ones(len(Z)), Z])   # матричные произведения — через einsum/sum (Accelerate даёт ложные предупреждения)
    pv = lam * np.ones(A.shape[1])
    pv[0] = 0.0
    b = np.zeros(A.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(A * b).sum(axis=1)))
        g = (A * (p - yb)[:, None]).sum(axis=0) + pv * b
        H = np.einsum("ni,nj,n->ij", A, A, p * (1 - p)) + np.diag(pv) + 1e-9 * np.eye(len(b))
        step = np.linalg.solve(H, g)
        b = b - step
        if np.abs(step).max() < 1e-10:
            break
    return b


def r3_prob(Zrow: np.ndarray, grp_hi: np.ndarray, grid: np.ndarray, label_shift: int = 1) -> np.ndarray:
    """p_d = P(группа строки d+1 = выс | z_d); обучение — пары t+label_shift <= d (метка известна). label_shift=0 — мутация (метка d+1 в обучении)."""
    out = np.full(len(grid), np.nan)
    for i, d in enumerate(grid):
        t = np.arange(0, d - label_shift + 1)
        ok = np.isfinite(Zrow[t]).all(axis=1) & np.isfinite(grp_hi[t + 1])
        if ok.sum() < 100 or len(np.unique(grp_hi[t + 1][ok])) < 2 or not np.isfinite(Zrow[d]).all():
            continue
        b = fit_logit(Zrow[t][ok], grp_hi[t + 1][ok])
        out[i] = 1 / (1 + np.exp(-(b[0] + (Zrow[d] * b[1:]).sum())))
    return out
