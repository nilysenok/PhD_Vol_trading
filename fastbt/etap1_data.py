"""Входы оценщика этапа 1 (К4, Ж-047): 5-мин бары, close Н6, реализованная σ, режим, события, тик — по тикерам.

Всё строго до `end` (исключительно); первый день после окна (Н5, строка b−1) — одна «свеча» из close Н6, без
внутридневных баров и σ. σ — только `sigma_real` (RV из 10-минуток до 18:50), `features_pit_v3.rv` не читается.
Тик (❓4): минимальная положительная разность цен 5-мин баров за год; на тикер берётся максимум по годам окна
(консервативнее: проход сквозь уровень на больший шаг). Данные читаются через `daily_1850.default_main()` и
`bars5m` (FASTBT_BARS1M).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fastbt import rezhim_e1, sigma_real, sobytiya_e1
from fastbt.bars5m import load_bars5m
from fastbt.exec_e_types import Day

START = pd.Timestamp("2017-01-01")                      # прогрев: σ, режим, состояние выходов
ROOT = Path(__file__).resolve().parent.parent
LIMIT = pd.Timestamp("2022-01-01")


def _one_bar(c: float) -> Day:
    a = np.array([c], float)
    return Day(a, a, a, a, False)


def tick_of(b: pd.DataFrame) -> float:
    """Шаг цены тикера: max по годам min положительной разности различных цен (OHLC), округление 1e-8."""
    out = []
    for _, g in b.groupby(pd.to_datetime(b["date"]).dt.year):
        p = np.unique(np.round(g[["open", "high", "low", "close"]].to_numpy(float).ravel(), 8))
        d = np.diff(p[np.isfinite(p)])
        out.append(float(d[d > 1e-9].min()) if (d > 1e-9).any() else 0.0)
    return max(out) if out else 0.0


def build_days(t: str, dates: np.ndarray, close: np.ndarray, s0: int, i1: int, end: pd.Timestamp, src=None):
    """Day по датам s0…i1 (дни без баров — pusto; i1 — одна свеча close Н6) и тик тикера."""
    b = load_bars5m(t, str(START.date()), str((end - pd.Timedelta(days=1)).date()), "s_predotkrytiem", src=src)
    b = b[~b["pusto"]]
    bd = pd.to_datetime(b["date"]).dt.normalize()
    grp = {d: g[["open", "high", "low", "close"]].to_numpy(float) for d, g in b.groupby(bd)}
    days, empty = [], np.empty(0)
    for i in range(s0, i1):
        a = grp.get(pd.Timestamp(dates[i]))
        days.append(Day(empty, empty, empty, empty, True) if a is None else Day(a[:, 0], a[:, 1], a[:, 2], a[:, 3], False))
    days.append(_one_bar(float(close[i1])))
    return days, tick_of(b)


def load_sigma(tickers, end: pd.Timestamp, main=None) -> dict:
    """{имя: {тикер: pd.Series(date → σ)}}; RV строго < end."""
    rv = sigma_real.load_rv(list(tickers), main)
    rv = rv[rv["date"] < end]
    out = {n: {} for n in sigma_real.NAMES}
    for t, g in rv.groupby("ticker"):
        r = g.set_index("date")["rv"].sort_index()
        for n, s in sigma_real.sigma_all(r).items():
            out[n][t] = s
    return out


def load_rezhim(end: pd.Timestamp, path=None) -> dict:
    """{имя: pd.Series(date → bool, True = вход разрешён)} по IMOEX < end."""
    d = rezhim_e1.imoex_daily(path)
    d = d[d["date"] < end].reset_index(drop=True)
    arr = rezhim_e1.rezhim_arrays(d["close"].to_numpy(float), d["sig_mkt"].to_numpy(float), d["nbars"].to_numpy())
    return {k: pd.Series(v, index=pd.DatetimeIndex(d["date"])) for k, v in arr.items()}


def load_events(dates_by_t: dict, end: pd.Timestamp) -> dict:
    """{имя: {тикер: bool по датам тикера}}: net, cbr (div/cbr_div — только после сверки ❓5, здесь не считаются)."""
    cbr = sobytiya_e1.load_cbr(ROOT / "data" / "calendar" / "cbr.csv")
    cbr = cbr[cbr["date"] < end]
    out = {"net": {}, "cbr": {}}
    for t, d in dates_by_t.items():
        ev = sobytiya_e1.sobytiya(d, [t], cbr)
        for k in out:
            out[k][t] = ev[k][t]
    return out


def align_f(series: pd.Series, dates) -> np.ndarray:
    """Ряд на даты тикера; нет значения — NaN (без подстановки, Д4)."""
    return series.reindex(pd.DatetimeIndex(dates)).to_numpy(float)


def align_b(series: pd.Series, dates) -> np.ndarray:
    """Флаг на даты тикера; нет значения — False (вход запрещён)."""
    return series.astype(float).reindex(pd.DatetimeIndex(dates)).fillna(0.0).to_numpy() > 0.5


def gate_trades(pos: np.ndarray, allow: np.ndarray) -> np.ndarray:
    """Режим/события/квантиль запрещают только ВХОД: сделка Sk целиком пропускается, если в день старта запрещено;
    открытая позиция не закрывается. pos ∈ {−1,0,1}, allow[d] = вход разрешён по close d."""
    pos = np.asarray(pos, float)
    prev = np.r_[0.0, pos[:-1]]
    start = (pos != 0) & (pos != prev)
    tid = np.cumsum(start) - 1
    ok = allow[np.flatnonzero(start)]
    keep = (pos != 0) & (tid >= 0)
    keep[keep] = ok[tid[keep]]
    return np.where(keep, pos, 0.0)
