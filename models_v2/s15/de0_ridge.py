"""DE0, G4: PIT-гребень (расширяющееся окно) — предсказуемость ln(RV/H) и дифференциала потерь по прошлым признакам M10 (S, Σ, z TCN-w5 V7).
Месячная переоценка: в первый день месяца модель учится на парах, ЗАКРЫТЫХ к этому дню (t⁺_j ≤ d_0), стандартизация и среднее — только по ним; λ = 100 объявлена (не подбирается).
R² вне выборки по году t⁺ = 1 − Σ(y − ŷ)² / Σ(y − ȳ_train)²: сравнение с безусловным средним прошлого. z года Y — энкодер Y на всех днях (оси z разных энкодеров не выровнены);
строки с NaN в признаках (нет z дня) в обучение и прогноз не идут."""
from __future__ import annotations

import numpy as np
import pandas as pd

LAM = 100.0
MIN_TRAIN = 250


def fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, lam: float = LAM) -> tuple:
    """Гребень на стандартизованных признаках (по train) → (прогноз, среднее train)."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    A, ym = (Xtr - mu) / sd, float(ytr.mean())
    # einsum вместо `@`: BLAS (Accelerate, macOS) даёт ложные RuntimeWarning в matmul; значения те же до 1e-13
    w = np.linalg.solve(np.einsum("ij,ik->jk", A, A) + lam * np.eye(A.shape[1]), np.einsum("ij,i->j", A, ytr - ym))
    return ym + np.einsum("ij,j->i", (Xte - mu) / sd, w), ym


def train_rows(tp, d0, first: int) -> np.ndarray:
    """Строки обучения в первый день месяца d0: пары, закрытые к d0 (t⁺ ≤ d0), и только до строки `first`."""
    return np.flatnonzero((np.asarray(tp) <= d0) & (np.arange(len(tp)) < first))


def pit_predict(X_of_year, y: np.ndarray, d: pd.DatetimeIndex, tp: pd.DatetimeIndex, lam: float = LAM, min_train: int = MIN_TRAIN) -> tuple:
    """X_of_year(Y) → (n, p) матрица признаков года Y для всех дат. -> (прогноз модели, среднее обучения) по датам; NaN — прогноза нет."""
    n = len(y)
    pred, base = np.full(n, np.nan), np.full(n, np.nan)
    for ym in sorted(set(zip(d.year, d.month))):
        sel = np.flatnonzero((d.year == ym[0]) & (d.month == ym[1]))
        X = X_of_year(ym[0])
        tr = train_rows(tp, d[sel[0]], sel[0])
        tr = tr[np.isfinite(X[tr]).all(1) & np.isfinite(y[tr])]
        if len(tr) < min_train:
            continue
        te = sel[np.isfinite(X[sel]).all(1)]
        pred[te], base[te] = fit_predict(X[tr], y[tr], X[te], lam)
    return pred, base


def pit_oos(X_of_year, y: np.ndarray, d: pd.DatetimeIndex, tp: pd.DatetimeIndex, lam: float = LAM, min_train: int = MIN_TRAIN) -> pd.DataFrame:
    """-> по году t⁺: n, sse_model, sse_mean, r2_oos."""
    pred, base = pit_predict(X_of_year, y, d, tp, lam, min_train)
    ok = np.isfinite(pred)
    rows = []
    for yr in sorted(set(tp.year[ok])):
        m = ok & (tp.year == yr)
        sm, sb = float(((y[m] - pred[m]) ** 2).sum()), float(((y[m] - base[m]) ** 2).sum())
        rows.append(dict(year=int(yr), n=int(m.sum()), sse_model=sm, sse_mean=sb, r2_oos=1.0 - sm / sb))
    return pd.DataFrame(rows)
