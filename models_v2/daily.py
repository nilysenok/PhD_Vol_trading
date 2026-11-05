"""Дневной ряд SBER из 10-минутных баров (Ж-197, О1–О3).

RV в десятичных долях (не в %^2). Только основная сессия: begin в [10:00, 18:40].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_BARS = 30
T_FROM, T_TO = "10:00", "18:40"
GAP = (pd.Timestamp("2022-02-28"), pd.Timestamp("2022-03-24"))  # остановка, включительно
W_WEEK, W_MONTH = 5, 22
SCALE = 1e4  # RV, BV, J в %^2
WEEKEND_FROM = pd.Timestamp("2025-03-01")  # сессии выходного дня — не основная сессия


def intraday_returns(b: pd.DataFrame) -> pd.DataFrame:
    """Баров основной сессии -> столбцы date, r, r_prev (r предыдущего бара дня).

    r_1 = ln(close_1/open_1); r_i = ln(close_i/close_{i-1}) — ночной гэп не входит.
    """
    b = b.sort_values("begin").reset_index(drop=True)
    day = b["begin"].dt.normalize()
    lc = np.log(b["close"].to_numpy(float))
    prev = pd.Series(lc).groupby(day).shift(1)
    r = lc - prev.to_numpy()
    first = prev.isna().to_numpy()
    r[first] = np.log(b["close"].to_numpy(float) / b["open"].to_numpy(float))[first]
    out = pd.DataFrame({"date": day, "r": r})
    out["r_prev"] = out.groupby("date")["r"].shift(1)
    return out


def main_session(bars: pd.DataFrame) -> pd.DataFrame:
    t = bars["begin"].dt.strftime("%H:%M")
    return bars[(t >= T_FROM) & (t <= T_TO)]


def _window_mean(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        out[w - 1:] = np.lib.stride_tricks.sliding_window_view(x, w).mean(axis=1)
    return out


def add_windows(d: pd.DataFrame) -> pd.DataFrame:
    """RV_w, RV_m — среднее уровней за 5/22 годных дня сегмента (включая d), без склейки через разрыв."""
    d = d.copy()
    d["seg"] = (d.index > GAP[1]).astype(int)
    d["rv_w"] = np.nan
    d["rv_m"] = np.nan
    for _, g in d.groupby("seg"):
        d.loc[g.index, "rv_w"] = _window_mean(g["rv"].to_numpy(), W_WEEK)
        d.loc[g.index, "rv_m"] = _window_mean(g["rv"].to_numpy(), W_MONTH)
    return d


def build_daily(bars: pd.DataFrame):
    """bars (begin, open, close, …) -> (daily, excluded).

    daily: индекс date; n_bars, rv, bv, j, seg, rv_w, rv_m. excluded: date, n_bars_main, reason.
    """
    all_days = bars["begin"].dt.normalize()
    main = main_session(bars)
    r = intraday_returns(main)
    g = r.groupby("date")
    n = g["r"].size()
    d = pd.DataFrame({"n_bars": n, "rv": (r["r"] ** 2).groupby(r["date"]).sum()})
    prod = (r["r"].abs() * r["r_prev"].abs()).groupby(r["date"]).sum()
    d["bv"] = np.pi / 2 * prod.reindex(d.index)
    d["j"] = np.maximum(d["rv"] - d["bv"], 0.0)
    d[["rv", "bv", "j"]] *= SCALE  # %^2 до любых логарифмов (решение дирижёра 05.11)
    ex = pd.DataFrame({"date": pd.Index(all_days.unique()).sort_values()})
    ex["n_bars_main"] = ex["date"].map(n).fillna(0).astype(int)
    ex["reason"] = ""
    ex.loc[ex["n_bars_main"] < MIN_BARS, "reason"] = "bars<30"
    ex.loc[(ex["date"].dt.dayofweek >= 5) & (ex["date"] >= WEEKEND_FROM), "reason"] = "weekend_2025"
    ex.loc[ex["date"].between(*GAP), "reason"] = "stop_2022"
    d = d.join(ex.set_index("date")["reason"], how="left")
    bad = d["reason"].ne("") | ~(d["rv"] > 0)
    ex.loc[ex["date"].isin(d.index[~(d["rv"] > 0) & d["reason"].eq("")]), "reason"] = "rv<=0"
    d = add_windows(d[~bad].drop(columns="reason"))
    return d, ex[ex["reason"] != ""].reset_index(drop=True)
