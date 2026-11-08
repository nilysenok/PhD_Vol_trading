"""V5-0, диагностика задним числом (G5–G7): МНК с t по Ньюи–Уэсту (лаг 5) по суммам по датам (кластер = дата, учёт корреляции между бумагами и во времени), частная корреляция. Путём прогноза не импортируется (тест AST)."""
from __future__ import annotations

import numpy as np
import pandas as pd

LAG = 5


def nw_ols(y, X, dates, lag: int = LAG) -> dict:
    """y (n), X (n×k, с константой, если нужна), dates (n) -> beta, t (НУ), r2, n. Ковариация: G_d = Σ_{i в дате d} x_i·e_i; S = Γ0 + Σ_l w_l(Γl + Γl')."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X, dates = y[ok], X[ok], np.asarray(dates)[ok]
    n, k = X.shape
    if n <= k + 10:
        return dict(beta=np.full(k, np.nan), t=np.full(k, np.nan), r2=np.nan, n=int(n))
    xtx_inv = np.linalg.pinv(X.T @ X)
    b = xtx_inv @ X.T @ y
    e = y - X @ b
    G = pd.DataFrame(X * e[:, None]).groupby(dates).sum().sort_index().to_numpy()
    S = G.T @ G
    for l in range(1, min(lag, len(G) - 1) + 1):
        A = G[l:].T @ G[:-l]
        S += (1.0 - l / (lag + 1.0)) * (A + A.T)
    V = xtx_inv @ S @ xtx_inv
    sst = float(((y - y.mean()) ** 2).sum()) if np.ptp(X[:, 0]) == 0 else float((y ** 2).sum())
    return dict(beta=b, t=b / np.sqrt(np.maximum(np.diag(V), 1e-300)), r2=float(1 - (e @ e) / sst) if sst > 0 else np.nan, n=int(n))


def demean(df: pd.DataFrame, cols: list, by: str = "ticker") -> pd.DataFrame:
    """Внутригрупповое центрирование (фиксированные эффекты бумаги)."""
    out = df.copy()
    out[cols] = df[cols] - df.groupby(by)[cols].transform("mean")
    return out


def partial(df: pd.DataFrame, ycol: str, xcol: str, controls: list, lag: int = LAG) -> dict:
    """Частная корреляция и t(x) при контроле `controls`, внутри бумаги (FE): остатки y и x по controls, corr остатков; t — МНК y ~ controls + x (НУ по датам); delta_r2 = R²(с x) − R²(без)."""
    d = df.dropna(subset=[ycol, xcol] + controls)
    d = demean(d, [ycol, xcol] + controls)
    if len(d) < 50:
        return dict(partial_corr=np.nan, beta=np.nan, t_nw=np.nan, delta_r2=np.nan, n=len(d))
    C = d[controls].to_numpy(float)
    y, x = d[ycol].to_numpy(float), d[xcol].to_numpy(float)
    if C.shape[1]:
        ry = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]
        rx = x - C @ np.linalg.lstsq(C, x, rcond=None)[0]
    else:
        ry, rx = y, x
    full = nw_ols(y, np.column_stack([x, C]), d["date"].to_numpy(), lag)
    base = nw_ols(y, C, d["date"].to_numpy(), lag) if C.shape[1] else dict(r2=0.0)
    return dict(partial_corr=float(np.corrcoef(ry, rx)[0, 1]), beta=float(full["beta"][0]), t_nw=float(full["t"][0]), delta_r2=float(full["r2"] - base["r2"]), n=full["n"])
