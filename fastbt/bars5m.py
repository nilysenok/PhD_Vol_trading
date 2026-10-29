"""Внутридневные 5-минутные бары (семья E) из 1-минуток data/regen/D25/bars1m/{T}_{Y}.csv.

Сетка и окно — как в moex_iss/src/rv5sub.rs: минута входит в окно по begin ∈ [from; to), 5-минутка
начинается на кратной 5 минуте от начала окна. Бары — OHLC (first/max/min/last), volume и value (суммы).
Пустая 5-минутка (в 1-минутках нет ни одной сделки) — явная строка: pusto=True, цены NaN, volume=value=0.
День без единой минуты в окне строк не даёт (праздник/нет торгов).

Окна ⚑ (не утверждены): "osnovnaya" 10:00–18:50 (106 пятиминуток), "s_predotkrytiem" 09:50–18:50 (108).
Период читается строго по запросу: файлы только лет start..end, строки обрезаются по датам.
По умолчанию 2017-01-01…2021-12-31; 2022+ только явным end.

Структурно пустая 5-минутка: 18:40–18:45 (последняя перед аукционом закрытия: идёт сбор заявок, сделка
аукциона печатается минутой 18:45; во всех днях и у всех 17 тикеров, не 14). Это ровно 1/106 = 0,94 %.
Дневной ряд ohlcv_daily_full совпадает с окном "s_predotkrytiem" только до 19.06.2020 (с 22.06.2020 в нём
вечерняя сессия, в 1-минутках её нет).

Кэш: <ROOT>/data/fastbt/bars5m/*.parquet, ключ = sha1(содержимое исходных CSV периода, код этого файла,
тикер, период, окно). Правка исходника или кода — новый ключ.
"""
from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(os.environ.get("FASTBT_BARS1M", ROOT / "data" / "regen" / "D25" / "bars1m"))
CACHE = Path(os.environ.get("FASTBT_BARS5M_CACHE", ROOT / "data" / "fastbt" / "bars5m"))
WINDOWS = {"osnovnaya": (10 * 60, 18 * 60 + 50), "s_predotkrytiem": (9 * 60 + 50, 18 * 60 + 50)}
STEP = 5
START, END = "2017-01-01", "2021-12-31"
COLS = ["ts", "date", "open", "high", "low", "close", "volume", "value", "pusto"]


def _clip(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    d = df["begin"].dt.normalize()
    return df[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))]


def _read(ticker: str, start: str, end: str, src: Path) -> pd.DataFrame:
    parts = []
    for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        p = src / f"{ticker}_{y}.csv"
        if p.exists():
            parts.append(pd.read_csv(p, parse_dates=["begin"]))
    if not parts:
        return pd.DataFrame(columns=["begin", "open", "high", "low", "close", "volume", "value"])
    return _clip(pd.concat(parts, ignore_index=True), start, end)


def aggregate(m1: pd.DataFrame, window: str = "osnovnaya") -> pd.DataFrame:
    """1-минутки (begin, open, high, low, close, volume, value) → 5-минутки с полной сеткой по дням."""
    f, t = WINDOWS[window]
    if m1.empty:
        return pd.DataFrame(columns=COLS)
    mod = m1["begin"].dt.hour * 60 + m1["begin"].dt.minute
    m = m1[(mod >= f) & (mod < t)].sort_values("begin")
    if m.empty:
        return pd.DataFrame(columns=COLS)
    m = m.assign(ts=m["begin"].dt.floor(f"{STEP}min"))
    g = m.groupby("ts").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                            close=("close", "last"), volume=("volume", "sum"), value=("value", "sum"))
    days = pd.DatetimeIndex(sorted(m["begin"].dt.normalize().unique()))
    offs = pd.to_timedelta(np.arange(f, t, STEP), unit="min")
    grid = (days.values[:, None] + offs.values[None, :]).ravel()
    out = g.reindex(pd.DatetimeIndex(grid, name="ts"))
    out["pusto"] = out["open"].isna()
    out[["volume", "value"]] = out[["volume", "value"]].fillna(0.0)
    out = out.reset_index()
    out["date"] = out["ts"].dt.normalize()
    return out[COLS]


def _key(ticker: str, start: str, end: str, window: str, src: Path) -> str:
    h = hashlib.sha1(Path(__file__).read_bytes())
    h.update(f"{ticker}|{start}|{end}|{window}".encode())
    for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        p = src / f"{ticker}_{y}.csv"
        h.update(p.read_bytes() if p.exists() else b"-")
    return f"{ticker}_{window}_{start}_{end}_{h.hexdigest()[:16]}"


def load_bars5m(ticker: str, start: str = START, end: str = END, window: str = "osnovnaya",
                src: Path | None = None, cache: Path | None = None) -> pd.DataFrame:
    """5-минутки тикера за [start; end]. Столбцы COLS; кэш на диске (FASTBT_NO_CACHE=1 — выключить)."""
    if window not in WINDOWS:
        raise ValueError(f"окно {window!r}: допустимо {sorted(WINDOWS)}")
    src, cache = Path(src or SRC), Path(cache or CACHE)
    use = os.environ.get("FASTBT_NO_CACHE", "") != "1"
    p = cache / f"{_key(ticker, start, end, window, src)}.parquet"
    if use and p.exists():
        return pd.read_parquet(p)
    df = aggregate(_read(ticker, start, end, src), window)
    if use:
        cache.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")       # Р-7: уникальный tmp (гонка двух процессов)
        df.to_parquet(tmp, index=False)
        os.replace(tmp, p)
    return df


def pusto_share(df: pd.DataFrame) -> float:
    return float(df["pusto"].mean()) if len(df) else float("nan")
