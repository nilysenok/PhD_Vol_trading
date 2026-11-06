"""Ж-214, V6: HARQ-J — реакция коэффициента дневной RV на качество измерения (Bollerslev–Patton–Quaedvlieg), не RQ как признак.
  ln RV_{t+1} = b0 + (b_d + b_dQ·q_t)·ln RV_d + b_w ln RV_w + b_m ln RV_m + b_J ln(1+J_d);  σ̂² = exp(f + s²/2) (OLS: H1, H4) или exp(f) (QLIKE-подгонка: H7).
q_t — z-оценка по дням s < t (>= 60, расширяющаяся): H1/H7 — q = z(ln √RQ_t), H4 — q = z(ln rv_rel,t), rv_rel = √(2 RQ/n)/RV. Пары с неопределённым q_t не входят.
Реакция R(q) = b_d + b_dQ·q (эластичность); ожидаемый знак b_dQ < 0 (BPQ). Коэффициенты — только закрытые пары (t+1 <= d)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import har as H
from models_v2.s8 import hs_qfit
from models_v2.s9 import hv2_world as W

COLS = ("b0", "b_d", "b_dQ", "b_w", "b_m", "b_J")


def q_series(w: W.World, kind: str, lead: int = 0) -> np.ndarray:
    """kind 'rq': z(ln √RQ); 'rel': z(ln rv_rel). Строка t. `lead` — МУТАНТ «день d+1»."""
    x = 0.5 * np.log(np.where(w.rq > 0, w.rq, np.nan)) if kind == "rq" else np.log(np.where(w.rel > 0, w.rel, np.nan))
    return W.zscore(x, lead=lead)


def design(w: W.World, q: np.ndarray) -> np.ndarray:
    """n x 5: [ln RV_d, q·ln RV_d, ln RV_w, ln RV_m, ln(1+J)] (строка t)."""
    X = w.p.X
    return np.column_stack([X[:, 0], q * X[:, 0], X[:, 1], X[:, 2], X[:, 3]])


def run(w: W.World, q: np.ndarray, qlike: bool = False) -> tuple:
    """-> (σ̂² на сетке w.el, коэффициенты n x 6 (b0, b_d, b_dQ, b_w, b_m, b_J), число пар)."""
    return fit_design(w, design(w, q), qlike)


def fit_design(w: W.World, Xd: np.ndarray, qlike: bool = False) -> tuple:
    """Лог-регрессия ln RV_{t+1} на столбцы Xd (+ константа) по парам t+1 <= d: OLS (σ̂² = exp(f + s²/2)) или QLIKE-подгонка (σ̂² = exp(f))."""
    s2, B, npair = np.empty(len(w.el)), np.empty((len(w.el), Xd.shape[1] + 1)), np.empty(len(w.el), int)
    for a, i in enumerate(w.el):
        idx = W.pairs_idx(w.p.seg, Xd, int(i))
        b, v, n = H.ols(Xd[idx], w.p.lnrv[idx + 1])
        x = np.r_[1.0, Xd[i]]
        if qlike:
            g0 = b.copy()
            g0[0] += v / 2
            b = hs_qfit.fit(np.column_stack([np.ones(n), Xd[idx]]), w.p.rv[idx + 1], np.zeros(n), gam0=g0, iters=60)
            s2[a] = float(np.exp(np.einsum("j,j->", b, x)))
        else:
            s2[a] = float(np.exp(np.einsum("j,j->", b, x) + v / 2))
        B[a], npair[a] = b, n
    return s2, B, npair


def reaction(model: str, w: W.World, B: np.ndarray) -> pd.DataFrame:
    """R(q) = b_d + b_dQ·q при q ∈ {−1; 0; +1} на последнюю дату сетки каждого года; среднее b_dQ за год и доля дней с b_dQ < 0."""
    yr = pd.DatetimeIndex(w.d).year.to_numpy()
    rows = []
    for Y in sorted(set(yr)):
        m = yr == Y
        b = B[np.flatnonzero(m)[-1]]
        rows.append({"model": model, "year": int(Y), "d_end": str(pd.Timestamp(w.d[np.flatnonzero(m)[-1]]).date()), "b_d": b[1], "b_dQ": b[2],
                     "R_q_minus1": b[1] - b[2], "R_q_0": b[1], "R_q_plus1": b[1] + b[2], "sign_b_dQ": "−" if b[2] < 0 else "+",
                     "b_dQ_mean_year": float(B[m, 2].mean()), "share_b_dQ_neg": float((B[m, 2] < 0).mean()), "n_days": int(m.sum())})
    return pd.DataFrame(rows)
