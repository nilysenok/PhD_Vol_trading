"""Постобработка прогнозов: Г3 (комбинация), Г4 (смесь по режиму), Г5 (перекалибровка). Ж-197 п. 3.

Все функции работают с массивами одной сетки дат, отсортированной по d. closed(k) = прогнозы с target_date <= d_k:
это первые m[k] элементов (прогноз k в них не входит — его цель ещё не известна).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from models_v2 import metrics

WIN = 252
W_GRID = np.round(np.arange(0, 11) / 10, 1)
CLIP = (0.7, 1.5)
Q_STRESS = 2 / 3
TIE = 1e-12      # ничья в Г3: разница QLIKE не больше TIE -> меньший w
MIN_Z = 30       # логистика Г4: минимум конечных z среди closed, иначе жёсткая смесь


def closed_count(d: np.ndarray, td: np.ndarray) -> np.ndarray:
    """m[k] = число прогнозов с target_date <= d[k] (d, td возрастают)."""
    return np.searchsorted(td, d, side="right")


def g3_weights(ln_b, ln_h, rv, m, win: int = WIN) -> np.ndarray:
    """w_k = argmin QLIKE смеси на последних win closed; до win closed — 0; ничья — меньший w."""
    w = np.zeros(len(rv))
    for k in range(len(rv)):
        if m[k] < win:
            continue
        s = slice(m[k] - win, m[k])
        ln = W_GRID[None, :] * ln_b[s, None] + (1 - W_GRID[None, :]) * ln_h[s, None]
        q = metrics.qlike(rv[s, None], np.exp(ln)).mean(axis=0)
        w[k] = W_GRID[int(np.flatnonzero(q <= q.min() + TIE)[0])]
    return w


def q_prior(x, q: float, win: int = WIN) -> np.ndarray:
    """Квантиль q по win предыдущим значениям (без текущего); до win значений — NaN."""
    out = np.full(len(x), np.nan)
    for k in range(win, len(x)):
        out[k] = np.quantile(x[k - win:k], q)
    return out


def q67_prior(rv_w, win: int = WIN) -> np.ndarray:
    """Квантиль 2/3 RV_w по win предыдущим датам сетки (без текущей); до win дат — NaN."""
    return q_prior(rv_w, Q_STRESS, win)


def b7_defense(rv_w, win: int = WIN) -> np.ndarray:
    """Б-7: RV_w,d выше 90-го процентиля RV_w по win предыдущим датам (без d); до win дат — ложь."""
    with np.errstate(invalid="ignore"):
        return rv_w > q_prior(rv_w, 0.9, win)


def b7_sigma2(s2_boost, s2_har, defense) -> np.ndarray:
    return np.where(defense, s2_har, s2_boost)


def stress(rvi_high, rv_w, q67) -> np.ndarray:
    """RVI выше прошлой медианы (NaN/после 2021 -> ложь) ИЛИ RV_w выше q67 (NaN -> ложь)."""
    hi = np.asarray(rvi_high, float)
    with np.errstate(invalid="ignore"):
        return (np.nan_to_num(hi, nan=0.0) > 0) | (rv_w > q67)


def g4_hard(ln_h, ln_c, st) -> np.ndarray:
    return np.where(st, ln_h, ln_c)


def _fit_ab(ln_h, ln_c, rv, z) -> tuple:
    ok = np.isfinite(z)
    if ok.sum() < MIN_Z:
        return None

    def loss(ab):
        pi = 1 / (1 + np.exp(-(ab[0] + ab[1] * z[ok])))
        return float(metrics.qlike(rv[ok], np.exp(pi * ln_h[ok] + (1 - pi) * ln_c[ok])).mean())

    r = minimize(loss, [0.0, 0.0], method="L-BFGS-B", bounds=[(-5, 5), (-5, 5)])
    return float(r.x[0]), float(r.x[1])


def g4_logit(ln_h, ln_c, rv, z, st, m, month, win: int = WIN) -> tuple:
    """-> (ln sigma2, a, b). (a, b) по closed на первый день сетки месяца; до win closed — жёсткая смесь."""
    out, a, b = g4_hard(ln_h, ln_c, st).astype(float), np.full(len(rv), np.nan), np.full(len(rv), np.nan)
    cur, last = None, None
    for k in range(len(rv)):
        if m[k] < win:
            continue
        if month[k] != last:
            last, mk = month[k], m[k]
            cur = _fit_ab(ln_h[:mk], ln_c[:mk], rv[:mk], z[:mk])
        if cur is None:
            continue
        a[k], b[k] = cur
        if np.isfinite(z[k]):
            pi = 1 / (1 + np.exp(-(cur[0] + cur[1] * z[k])))
            out[k] = pi * ln_h[k] + (1 - pi) * ln_c[k]
    return out, a, b


def ewma_k(rv, s2, m, halflife: float, clip=CLIP) -> np.ndarray:
    """k_k = clip(взвешенное среднее RV/σ̂² по closed, вес 0,5^(возраст/halflife), возраст 0 — самый свежий)."""
    r, k = rv / s2, np.ones(len(rv))
    for j in range(len(rv)):
        if m[j] == 0:
            continue
        w = 0.5 ** (np.arange(m[j] - 1, -1, -1) / halflife)
        k[j] = np.clip((w * r[:m[j]]).sum() / w.sum(), clip[0], clip[1])
    return k


def g5(s2, k, st=None) -> np.ndarray:
    """σ̂²' = k·σ̂²; при st — только в стресс."""
    return s2 * (k if st is None else np.where(st, k, 1.0))
