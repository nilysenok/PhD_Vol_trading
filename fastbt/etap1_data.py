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

from fastbt import rezhim_e1, sigma_real, sobytiya_e1, zona
from fastbt.bars5m import load_bars5m
from fastbt.exec_e_types import Day

START = pd.Timestamp("2017-01-01")                      # прогрев: σ, режим, состояние выходов
ROOT = Path(__file__).resolve().parent.parent
LIMIT = pd.Timestamp("2022-01-01")
START_Z2 = pd.Timestamp("2021-01-01")                  # реверс раунд 1: тест-1 2022–2024, прогрев 2021 (разработка, разрешён)
START_V1 = pd.Timestamp("2015-01-01")                 # К-в1: train 2016–2019, 2015 — прогрев (в метрики не входит)


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


def _minutes(b: pd.DataFrame) -> np.ndarray:
    """Начало каждого бара (ts), минуты от полуночи (Л-3); без баров — пустой массив, колонка ts не читается."""
    if not len(b):
        return np.empty(0, np.int64)
    ts = pd.to_datetime(b["ts"])
    return (ts.dt.hour * 60 + ts.dt.minute).to_numpy(np.int64)


def build_days(t: str, dates: np.ndarray, close: np.ndarray, s0: int, i1: int, end: pd.Timestamp, src=None, start: pd.Timestamp | None = None,
               last_year: int | None = None):
    """Day по датам s0…i1 (дни без баров — pusto; i1 — одна свеча close Н6) и тик тикера.
    start — начало прогрева (по умолчанию START = 2017-01-01; В1 — START_V1). Чтение ≥ LIMIT запрещено;
    last_year — последний год train: end > 1 января last_year+1 запрещено (В1: 2019 — validation 2020–2021 закрыта)."""
    lim = zona.end()                                       # 2022-01-01 в dev (= LIMIT); зона Z2 — 2025-01-01
    cap = lim if last_year is None else min(lim, pd.Timestamp(f"{last_year + 1}-01-01"))
    if end > cap:
        raise PermissionError(f"чтение баров до {end.date()}: граница периода {cap.date()}")
    b = load_bars5m(t, str((start or START).date()), str((end - pd.Timedelta(days=1)).date()), "s_predotkrytiem", src=src)
    b = b[~b["pusto"]]
    bd = pd.to_datetime(b["date"]).dt.normalize()
    b = b.assign(_t=_minutes(b))
    grp = {d: g[["open", "high", "low", "close"]].to_numpy(float) for d, g in b.groupby(bd)}
    grt = {d: g["_t"].to_numpy(np.int64) for d, g in b.groupby(bd)}
    days, empty = [], np.empty(0)
    for i in range(s0, i1):
        key = pd.Timestamp(dates[i])
        a = grp.get(key)
        days.append(Day(empty, empty, empty, empty, True) if a is None else Day(a[:, 0], a[:, 1], a[:, 2], a[:, 3], False, t=grt[key]))
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


def psr_dsr(x: np.ndarray, M: np.ndarray, n_all: int) -> dict:
    """Справочно (Ж-044): T = дней; var SR — по столбцам M, N = все посчитанные точки."""
    from fastbt import sweep_drivers as sd                                            # noqa: PLC0415
    sr = lambda v: float(v.mean() / v.std(ddof=1)) if v.std(ddof=1) > 0 else 0.0     # noqa: E731
    if x.std(ddof=1) == 0:
        return dict(T=len(x), psr=float("nan"), dsr=float("nan"), N_dsr=n_all)
    z = (x - x.mean()) / x.std(ddof=1)
    tr = [sr(M[:, j]) for j in range(M.shape[1])]
    star = sd.expected_max_sharpe(n_all, float(np.var(tr, ddof=1)) if len(tr) > 1 else 0.0)
    a = (len(x), float((z ** 3).mean()), float((z ** 4).mean()))
    return dict(T=len(x), psr=sd.psr(sr(x), 0.0, *a), dsr=sd.psr(sr(x), star, *a), N_dsr=n_all)


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
