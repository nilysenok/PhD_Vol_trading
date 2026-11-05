"""Дневной ряд SBER из 10-минутных баров (Ж-197, О1–О3).

RV в десятичных долях (не в %^2). Только основная сессия: begin в [10:00, 18:40].
"""
from __future__ import annotations

from pathlib import Path

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


IMOEX_PATH = Path(__file__).resolve().parents[1] / "data/iss_check/index_10m/IMOEX.parquet"
_IDAYS: set | None = None


def index_session_days() -> set:
    """Даты, в которые у IMOEX есть хотя бы один бар основной сессии (кэш на процесс)."""
    global _IDAYS
    if _IDAYS is None:
        im = main_session(pd.read_parquet(IMOEX_PATH, columns=["begin"]))
        _IDAYS = set(im["begin"].dt.normalize())
    return _IDAYS


def build_daily(bars: pd.DataFrame, index_days="auto"):
    """bars (begin, open, close, …) -> (daily, excluded).

    daily: индекс date; n_bars, rv, bv, j, seg, rv_w, rv_m. excluded: date, n_bars_main, reason.
    index_days: даты работы IMOEX; с 01.03.2025 день без сессии IMOEX исключается (nonworking_2025, ↺ 05.11).
    "auto" — из файла IMOEX (читается, только если в барах есть даты с 01.03.2025); None — правило выключено (для самого IMOEX).
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
    late = ex["date"] >= WEEKEND_FROM
    if index_days is not None and late.any():
        idays = index_session_days() if isinstance(index_days, str) else set(index_days)
        ex.loc[late & ~ex["date"].isin(idays), "reason"] = "nonworking_2025"
    ex.loc[(ex["date"].dt.dayofweek >= 5) & late, "reason"] = "weekend_2025"
    ex.loc[ex["date"].between(*GAP), "reason"] = "stop_2022"
    d = d.join(ex.set_index("date")["reason"], how="left")
    bad = d["reason"].ne("") | ~(d["rv"] > 0)
    ex.loc[ex["date"].isin(d.index[~(d["rv"] > 0) & d["reason"].eq("")]), "reason"] = "rv<=0"
    d = add_windows(d[~bad].drop(columns="reason"))
    return d, ex[ex["reason"] != ""].reset_index(drop=True)
