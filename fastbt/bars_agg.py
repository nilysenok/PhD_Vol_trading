"""Агрегатор баров для проверки таймфреймов (Ж-098, Ж-100): готовые 10-минутки → 30м / 1ч / 2ч → день → неделя.

Чтение 10-минуток — готовое `fastbt.daily_1850.read_bars` (candles_10m/{T}.parquet, фильтр по времени на чтении pyarrow,
конец — аргумент, ≥ 2020 не читается). Здесь только то, чего там нет: окно сигналов [10:00; 18:50) (утренний бар
предоткрытия 09:50 не входит: вход d+1 — по open 10:00, Ж-086), слот бара, укрупнение 30м/1ч/2ч, дневной и недельный бар.
Последний бар 1ч / 2ч дня — неполный (18:00–18:50): это «последний бар до 18:50». Дневной close — последний бар до 18:50.
OHLCV: open — первый, high — max, low — min, close — последний, volume — сумма. Недельный бар — из дневных
(ISO-неделя, date = последний торговый день недели). Колонки: date, [slot], open, high, low, close, volume.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from fastbt import daily_1850 as d18

OPEN_MIN, CLOSE_MIN, STEP = 10 * 60, 18 * 60 + 50, 10
SLOTS = (CLOSE_MIN - OPEN_MIN + STEP - 1) // STEP  # 53: слот 0 = 10:00, слот 52 = 18:40
K = {"10m": 1, "30m": 3, "1h": 6, "2h": 12}
_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def prep_10m(bars: pd.DataFrame) -> pd.DataFrame:
    """10-минутки (timestamp, OHLCV) → (date, slot, OHLCV); всё с началом вне [10:00; 18:50) отброшено."""
    b = bars.sort_values("timestamp")
    mod = b["timestamp"].dt.hour * 60 + b["timestamp"].dt.minute
    b = b[(mod >= OPEN_MIN) & (mod < CLOSE_MIN)]
    mod = b["timestamp"].dt.hour * 60 + b["timestamp"].dt.minute
    b = b.assign(date=b["timestamp"].dt.normalize(), slot=(mod - OPEN_MIN) // STEP)
    return b[["date", "slot", *_AGG]].reset_index(drop=True)


def load_10m(ticker: str, end: str, main: Path | None = None) -> pd.DataFrame:
    """10-минутки тикера до end (исключительно) через готовый read_bars; окно сигналов применено."""
    p = (main or d18.default_main()) / "moex_discovery" / "data" / "raw" / "candles_10m" / f"{ticker}.parquet"
    return prep_10m(d18.read_bars(p, pd.Timestamp(end)))


def resample(b10: pd.DataFrame, k: int) -> pd.DataFrame:
    """10-минутки → бары в k раз крупнее внутри дня (группа = slot // k); слоты ≥ 18:50 не участвуют."""
    b = b10[b10["slot"] < SLOTS]
    return b.assign(slot=b["slot"] // k).groupby(["date", "slot"]).agg(_AGG).reset_index()


def daily(b10: pd.DataFrame) -> pd.DataFrame:
    """10-минутки → дневные бары (open первого бара, close последнего бара до 18:50)."""
    return b10[b10["slot"] < SLOTS].groupby("date").agg(_AGG).reset_index()


def weekly(d: pd.DataFrame) -> pd.DataFrame:
    """Дневные бары → недельные по ISO-неделе; date — последний торговый день недели."""
    iso = d["date"].dt.isocalendar()
    g = d.groupby([iso["year"].values, iso["week"].values])
    return g.agg({"date": "last", **_AGG}).reset_index(drop=True)
