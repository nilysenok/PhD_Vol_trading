"""Волна D, поток LGBM: причинные режимные ряды (RD3) — стресс Г4 (готовый ряд p3), квинтиль прошлой RV_w, переход. Значение в строке t — только данные ≤ t."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s5 import common
from models_v2.s6.lgbm.job import Spec

MIN_Q, TRANS_K = 60, 5


def stress_rows(dates) -> np.ndarray:
    """Стресс Г4 по строкам F (1/0, NaN до 30.12.2016, где ряда нет). Ряд p3 общий для потоков, здесь не пересчитывается."""
    st = pd.read_parquet(common.OUT / "stress_p3.parquet").set_index("d")["stress"].astype(float)
    return st.reindex(pd.DatetimeIndex(dates)).to_numpy(float)


def rv_w_rows(dates) -> np.ndarray:
    return pd.read_parquet(common.OUT / "daily_p1.parquet")["rv_w"].reindex(pd.DatetimeIndex(dates)).to_numpy(float)


def quintile_pit(x, min_n: int = MIN_Q) -> np.ndarray:
    """Квинтиль 0…4 значения x[t] среди x[≤ t] (расширяющийся ранг: доля строго меньших); раньше min_n наблюдений — NaN."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    for t in range(len(x)):
        h = x[:t + 1]
        h = h[np.isfinite(h)]
        if len(h) >= min_n and np.isfinite(x[t]):
            out[t] = min(4, int(5 * (h < x[t]).mean()))
    return out


def transition_pit(st, k: int = TRANS_K) -> np.ndarray:
    """1, если стресс менялся в строках t−k+1…t (сравнение с предыдущей строкой), иначе 0; NaN, где стресса нет."""
    st = np.asarray(st, float)
    flip = np.r_[np.nan, (st[1:] != st[:-1]).astype(float)]
    flip[np.isnan(st) | np.isnan(np.r_[np.nan, st[:-1]])] = np.nan
    tr = pd.Series(flip).rolling(k, min_periods=1).max().to_numpy(float)
    tr[np.isnan(st)] = np.nan
    return tr


def flip_on_grid(st_rows, grid) -> np.ndarray:
    """Смена режима на дате сетки относительно предыдущей даты сетки (для adapt); NaN считается отсутствием смены."""
    s = np.asarray(st_rows, float)[np.asarray(grid, int)]
    f = np.r_[False, (s[1:] != s[:-1]) & np.isfinite(s[1:]) & np.isfinite(s[:-1])]
    return f


R1 = {"R1a_stress": ("stress",), "R1b_quintile": ("q5",), "R1c_transition": ("trans",), "R1d_all": ("stress", "q5", "trans")}


def feature_cols(dates) -> dict:
    st = stress_rows(dates)
    return {"stress": st, "q5": quintile_pit(rv_w_rows(dates)), "trans": transition_pit(st)}


def with_regime(sp: Spec, dates, which: tuple, name: str) -> Spec:
    """База + режимные столбцы в конец X (NaN допустимы: LightGBM). Пары требуют конечности только столбцов 0–3."""
    cols = feature_cols(dates)
    X = np.column_stack([sp.X] + [cols[w] for w in which])
    return Spec(name, sp.kind, X, sp.y, sp.off, sp.wt)
