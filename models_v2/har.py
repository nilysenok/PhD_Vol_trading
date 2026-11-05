"""Панель признаков, расширяющееся окно, OLS HAR / HAR-J, эталоны (Ж-197, О4–О6).

Прогноз с дня i (индекс строки дневного ряда) использует строки t <= i; пары (x_t, y_{t+1}) — только t+1 <= i
и только внутри одного сегмента (без склейки через разрыв 2022).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

COLS_HAR = (0, 1, 2)       # ln RV_d, ln RV_w, ln RV_m
COLS_HARJ = (0, 1, 2, 3)   # + ln(1+J_d)
COL_NAMES = ("ln_rv", "ln_rv_w", "ln_rv_m", "ln1p_j")


@dataclass
class Panel:
    dates: np.ndarray   # datetime64[ns]
    seg: np.ndarray
    rv: np.ndarray
    X: np.ndarray       # n x 4, NaN где окно не набрано

    @property
    def lnrv(self) -> np.ndarray:
        return np.log(self.rv)


def make_panel(daily: pd.DataFrame) -> Panel:
    X = np.column_stack([np.log(daily["rv"]), np.log(daily["rv_w"]), np.log(daily["rv_m"]),
                         np.log1p(daily["j"])])
    return Panel(daily.index.to_numpy(), daily["seg"].to_numpy(), daily["rv"].to_numpy(float), X)


def ols(X: np.ndarray, y: np.ndarray):
    """OLS с константой: (beta, s2 = SSR/(n-k), n). Отказ при n <= k."""
    n, k = len(y), X.shape[1] + 1
    if n <= k:
        raise ValueError("мало наблюдений для OLS: n=%d, k=%d" % (n, k))
    A = np.column_stack([np.ones(n), X])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    res = y - (A * beta).sum(axis=1)  # без matmul: Accelerate на macOS даёт ложные RuntimeWarning
    return beta, float((res * res).sum()) / (n - k), n


def pairs(p: Panel, i: int, cols, upto=None):
    """Обучающие пары для прогноза с i: t+1 <= upto (по умолчанию i), один сегмент, признаки конечны."""
    u = i if upto is None else upto
    t = np.arange(0, u)
    ok = (p.seg[t] == p.seg[t + 1]) & np.isfinite(p.X[t][:, list(cols)]).all(axis=1)
    t = t[ok]
    return p.X[t][:, list(cols)], p.lnrv[t + 1]


def fit_window(p: Panel, i: int, cols):
    Xt, yt = pairs(p, i, cols)
    return ols(Xt, yt)


def predict(beta: np.ndarray, x: np.ndarray, s2: float):
    """(sigma2_a, sigma2_b): exp(f) и exp(f + s2/2)."""
    f = float(beta[0] + beta[1:] @ x)
    return float(np.exp(f)), float(np.exp(f + s2 / 2))


def forecast_at(p: Panel, i: int) -> dict:
    """Все 7 моделей для прогноза с строки i (на следующий годный день)."""
    out = {"rw": float(p.rv[i]), "rv_w": float(np.exp(p.X[i, 1])), "rv_m": float(np.exp(p.X[i, 2]))}
    for name, cols in (("har", COLS_HAR), ("harj", COLS_HARJ)):
        beta, s2, _ = fit_window(p, i, cols)
        out[name + "_a"], out[name + "_b"] = predict(beta, p.X[i, list(cols)], s2)
    return out


def eligible(p: Panel, first_target: str = "2017-01-03") -> np.ndarray:
    """Сетка дат: i, у которого есть цель (следующая строка того же сегмента), конечные признаки,
    целевая дата >= first_target. Общая для всех моделей."""
    n = len(p.rv)
    i = np.arange(n - 1)
    ok = (p.seg[i] == p.seg[i + 1]) & np.isfinite(p.X[i]).all(axis=1)
    ok &= p.dates[i + 1] >= np.datetime64(first_target)
    return i[ok]


def forecast_table(p: Panel, first_target: str = "2017-01-03") -> pd.DataFrame:
    rows = []
    for i in eligible(p, first_target):
        for m, s2 in forecast_at(p, int(i)).items():
            rows.append((p.dates[i], p.dates[i + 1], m, s2, p.rv[i + 1]))
    return pd.DataFrame(rows, columns=["d", "target_date", "model", "sigma2", "rv"])
