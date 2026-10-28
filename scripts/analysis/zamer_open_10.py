"""Справочный замер Ж-086 п. 4 (данные ≤ 2019): насколько новая цена выхода отличается от старой.

|open бара 10:00 − open первого бара дня| и |open бара 15:00 − close бара 14:55|, в % от старой цены.
3 тикера × 3 даты (2016, 2018, 2019). 5-минутные бары — из 1-минуток окна 09:50–18:50 (как fastbt/bars5m).
Запуск: python3 scripts/analysis/zamer_open_10.py [каталог bars1m]; без аргументов — самопроверка на синтетике.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(os.environ.get("FASTBT_BARS1M", ROOT / "data" / "regen" / "D25" / "bars1m"))
TICKERS = ("SBER", "LKOH", "MTSS")
DATES = ("2016-03-15", "2018-03-14", "2019-03-13")        # только ≤ 2019: данные ≥ 2020 не читаются
LIMIT = pd.Timestamp("2020-01-01")


def bars5(m1: pd.DataFrame) -> pd.DataFrame:
    """5-минутные бары окна 09:50–18:50 из 1-минуток одного дня: open — первой минуты, close — последней."""
    m1 = m1.set_index("begin").sort_index()
    t = m1.index.hour * 60 + m1.index.minute
    m1 = m1[(t >= 590) & (t < 1130)]
    return m1.resample("5min").agg({"open": "first", "close": "last"}).dropna()


def zamer(b: pd.DataFrame) -> tuple[float, float]:
    """(|o10 − o_первого| / o_первого, |o15 − c14:55| / c14:55) в %; нет нужного бара — ValueError."""
    day = b.index[0].normalize()
    o10, o15 = b.loc[day + pd.Timedelta(hours=10), "open"], b.loc[day + pd.Timedelta(hours=15), "open"]
    c1455 = b.loc[day + pd.Timedelta(hours=14, minutes=55), "close"]
    return abs(o10 - b["open"].iloc[0]) / b["open"].iloc[0] * 100, abs(o15 - c1455) / c1455 * 100


def run(src: Path) -> None:
    print("тикер  дата        |o10 − o_первого|, %   |o15 − c14:55|, %   (первый бар)")
    for tk in TICKERS:
        for ds in DATES:
            if pd.Timestamp(ds) >= LIMIT:
                raise PermissionError(ds)
            df = pd.read_csv(src / f"{tk}_{ds[:4]}.csv", parse_dates=["begin"])
            m1 = df[df["begin"].dt.strftime("%Y-%m-%d") == ds]
            if m1.empty:
                print(f"{tk:5}  {ds}  нет торгов")
                continue
            b = bars5(m1)
            a, c = zamer(b)
            print(f"{tk:5}  {ds}  {a:8.3f}               {c:8.3f}            {b.index[0].strftime('%H:%M')}")


def selftest() -> None:
    idx = pd.date_range("2016-03-15 09:55", "2016-03-15 18:45", freq="5min")
    m1 = pd.DataFrame({"begin": idx, "open": 100.0, "close": 100.0})
    m1.loc[m1["begin"] == pd.Timestamp("2016-03-15 09:55"), "open"] = 99.0          # первый бар 09:55
    m1.loc[m1["begin"] == pd.Timestamp("2016-03-15 10:00"), "open"] = 101.0
    m1.loc[m1["begin"] == pd.Timestamp("2016-03-15 14:55"), "close"] = 100.0
    m1.loc[m1["begin"] == pd.Timestamp("2016-03-15 15:00"), "open"] = 102.0
    a, c = zamer(bars5(m1))
    assert abs(a - 2 / 99 * 100) < 1e-9 and abs(c - 2.0) < 1e-9, (a, c)
    print("самопроверка: ок")


if __name__ == "__main__":
    selftest() if len(sys.argv) == 1 else run(Path(sys.argv[1]))
