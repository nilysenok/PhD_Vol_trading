"""V7-1a-E: шаг E0 — гейт новизны. R² каждой компоненты z на опорных признаках X_ref (S, Σ, ln H, ℓ = ln(M/H) — столбцы собирает стадия 1b).
PIT-регрессия: гребень (λ = 1 на стандартизованных по прошлому X_ref, свободный член), расширяющееся окно; прогноз на дни месяца — по дням строго до первого дня месяца,
помесячная переоценка. R² = 1 − Σ(z − ẑ)² / Σ(z − z̄_прошлого)² по дням окна оценки (внепробный: «частный» вклад — сколько z объясняется опорными признаками).
Порог плана: R² ≥ 0,9 по ВСЕМ компонентам на 2018–2021 ⇒ энкодер новой информации не несёт (E0 стоп)."""
from __future__ import annotations

import numpy as np
import pandas as pd

THRESH, LAM, MIN_PAST = 0.9, 1.0, 60


def _ridge_pred(Xp, zp, Xe, lam):
    mu, sd = Xp.mean(0), Xp.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    A, zb = (Xp - mu) / sd, zp.mean(0)
    es = lambda a, b: np.einsum("ij,ik->jk", a, b)       # AᵀB; einsum — вместо `@` (ложные предупреждения Accelerate numpy 2.0 на macOS)
    beta = np.linalg.solve(es(A, A) + lam * np.eye(A.shape[1]), es(A, zp - zb))
    return zb + np.einsum("ij,jk->ik", (Xe - mu) / sd, beta), zb


def novelty_gate(z: np.ndarray, X_ref: np.ndarray, dates, start="2018-01-01", end="2021-12-31",
                 lam: float = LAM, min_past: int = MIN_PAST, thr: float = THRESH) -> dict:
    """-> dict(r2 (d_z,), no_new_info bool, pred (N, d_z) с NaN вне окна, n_eval)."""
    d = pd.DatetimeIndex(dates)
    if not d.is_monotonic_increasing:
        raise ValueError("даты должны идти по возрастанию")
    z, X = np.asarray(z, float), np.asarray(X_ref, float)
    z = z[:, None] if z.ndim == 1 else z
    ev = (d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))
    pred, base = np.full(z.shape, np.nan), np.full(z.shape, np.nan)
    month = d.to_period("M")
    for m in month[ev].unique():
        cur = np.flatnonzero(ev & (month == m))
        past = np.flatnonzero(d < d[cur[0]])
        if len(past) < min_past:
            continue
        pred[cur], base[cur] = _ridge_pred(X[past], z[past], X[cur], lam)
    ok = np.isfinite(pred[:, 0])
    if not ok.any():
        raise ValueError("в окне оценки нет дат с достаточным прошлым")
    sse, sst = ((z[ok] - pred[ok]) ** 2).sum(0), ((z[ok] - base[ok]) ** 2).sum(0)
    r2 = 1.0 - sse / np.where(sst > 0, sst, np.nan)
    return dict(r2=r2, no_new_info=bool(np.all(r2 >= thr)), pred=pred, n_eval=int(ok.sum()))
