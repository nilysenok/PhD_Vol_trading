"""Ж-214, V5 Q4/Q5.
Q4 `loglin_qlike_decay`: A0-спецификация (4 φ: ln RV_d, ln RV_w, ln RV_m, ln(1+J)), σ̂² = exp(Φγ) (без s²/2), γ — минимум ВЗВЕШЕННОГО QLIKE на закрытых парах,
w_t = 0,5^{(i−1−t)/h} (возраст в строках ряда), полураспад h ∈ {250; 500; 1000; ∞}; h для дня d — по среднему QLIKE закрытых прогнозов (target_date <= d, расширяющееся,
>= 252 закрытых, иначе h = ∞); ничья (|Δ| <= 1e-12) — больший h. Весь выбор = 1 trial.
Q5 `scale_c_pit`: коэффициенты A0, σ̂² = c_d·exp(f); c_d = mean RV_{j+1}/exp(f_j) по закрытым j (>= 252, иначе exp(f + s²/2)) — QLIKE-оптимальная шкала при фиксированной форме."""
from __future__ import annotations

import numpy as np

from models_v2 import metrics
from models_v2.s8 import hs_pit
from models_v2.s9 import hv2_a0 as A0, hv2_world as W

HL = (250.0, 500.0, 1000.0, np.inf)
MIN_CLOSED = 252


def phi(w: W.World) -> np.ndarray:
    return np.column_stack([np.ones(len(w.p.rv)), w.p.X])


def wfit(Phi: np.ndarray, y: np.ndarray, wt: np.ndarray, iters: int = 60, tol: float = 1e-10) -> np.ndarray:
    """argmin Σ wt_t (η_t + y_t e^{−η_t}) / Σ wt, η = Φγ (Newton с обратным шагом); старт — WLS в логарифмах + оптимальный сдвиг свободного члена."""
    wt = wt / wt.sum()
    sw = np.sqrt(wt)
    gam = np.linalg.lstsq(Phi * sw[:, None], np.log(y) * sw, rcond=None)[0]
    gam[0] += np.log(np.sum(wt * y * np.exp(-np.einsum("ij,j->i", Phi, gam))))

    def loss(g):
        eta = np.einsum("ij,j->i", Phi, g)
        return float(np.sum(wt * (eta + y * np.exp(-eta))))

    for _ in range(iters):
        r = y * np.exp(-np.einsum("ij,j->i", Phi, gam))
        grad = np.einsum("ij,i->j", Phi, wt * (1.0 - r))
        Hs = np.einsum("ij,i,ik->jk", Phi, wt * r, Phi) + 1e-12 * np.eye(Phi.shape[1])
        step = np.linalg.solve(Hs, grad)
        t, l0 = 1.0, loss(gam)
        while loss(gam - t * step) > l0 and t > 1e-8:
            t /= 2
        gam = gam - t * step
        if np.max(np.abs(t * step)) < tol:
            break
    return gam


def run_hl(w: W.World, h: float) -> np.ndarray:
    """σ̂² на сетке при фиксированном полураспаде h."""
    Phi, out = phi(w), np.empty(len(w.el))
    for a, i in enumerate(w.el):
        idx = W.pairs_idx(w.p.seg, Phi[:, 1:], int(i))
        wt = np.ones(len(idx)) if np.isinf(h) else 0.5 ** ((int(i) - 1 - idx) / h)
        gam = wfit(Phi[idx], w.p.rv[idx + 1], wt)
        out[a] = float(np.exp(np.einsum("j,j->", gam, Phi[i])))
    return out


def choose_hl(S: dict, rv_t, d, td, extra: int = 0) -> tuple:
    """S: {h: σ̂² сетки}. -> (σ̂² выбранного h по дням, индекс в HL). `extra` > 0 — МУТАНТ «закрыто + extra»."""
    m = hs_pit.closed_count(d, td, extra)
    cum = {h: np.concatenate([[0.0], np.cumsum(metrics.qlike(rv_t, S[h]))]) for h in HL}
    out, pick = np.empty(len(d)), np.empty(len(d), int)
    for k in range(len(d)):
        best = HL[-1]
        if m[k] >= MIN_CLOSED:
            mean = {h: cum[h][m[k]] / m[k] for h in HL}
            for h in HL[::-1][1:]:
                if mean[h] < mean[best] - 1e-12:
                    best = h
        out[k], pick[k] = S[best][k], HL.index(best)
    return out, pick


def run_decay(w: W.World, extra: int = 0) -> tuple:
    S = {h: run_hl(w, h) for h in HL}
    s2, pick = choose_hl(S, w.target, w.d, w.td, extra)
    return s2, pick, S


def scale_c(sa, sb, rv_t, d, td, extra: int = 0) -> np.ndarray:
    """σ̂²_a·c_d (c_d — среднее RV/σ̂²_a по закрытым >= 252), иначе σ̂²_b."""
    m = hs_pit.closed_count(d, td, extra)
    cs = np.concatenate([[0.0], np.cumsum(np.asarray(rv_t, float) / np.asarray(sa, float))])
    out = np.array(sb, float, copy=True)
    ok = m >= MIN_CLOSED
    out[ok] = np.asarray(sa, float)[ok] * cs[m[ok]] / m[ok]
    return out


def run_scale(w: W.World, extra: int = 0) -> tuple:
    s = A0.harj_series(w.p, w.el)
    return scale_c(s["sigma2_a"].to_numpy(), s["sigma2_b"].to_numpy(), w.target, w.d, w.td, extra), s
