"""Н6 (Ж-029): дневной ряд до 18:50 — вечерние бары не меняют OHLCV дня; данные только ≤ 2021."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from fastbt import daily_1850 as D

MAIN = D.default_main()
CANDLES = MAIN / "moex_discovery" / "data" / "raw" / "candles_10m"
OLD = MAIN / "moex_volatility" / "results" / "final" / "strategies" / "data" / "ohlcv_daily_full.parquet"
DEV_END = pd.Timestamp("2021-12-31")


def synth(evening: bool) -> pd.DataFrame:
    rows = [("2020-07-01 09:50", 100, 101, 99, 100.5, 10), ("2020-07-01 10:00", 100.5, 103, 100, 102, 20),
            ("2020-07-01 18:40", 102, 102.5, 101, 101.5, 30), ("2020-07-01 18:45", 101.5, 102, 101.2, 101.8, 40)]
    if evening:                       # подложенные вечерние бары с другой ценой, шире диапазона
        rows += [("2020-07-01 18:55", 101.8, 150, 50, 140, 1000), ("2020-07-01 23:40", 140, 200, 10, 5, 5000)]
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"]).assign(
        timestamp=lambda x: pd.to_datetime(x["timestamp"]))


def check_evening_invariant(cut_min=D.CUT_MIN):
    a = D.aggregate(synth(False), "X", cut_min)
    b = D.aggregate(synth(True), "X", cut_min)
    pd.testing.assert_frame_equal(a, b)
    r = b.iloc[0]
    assert (r["open"], r["high"], r["low"], r["close"], r["volume"]) == (100, 103, 99, 101.8, 100)


def test_evening_bars_do_not_change_day():
    check_evening_invariant()


def test_mutation_boundary_2359_is_caught():
    with pytest.raises(AssertionError):
        check_evening_invariant(cut_min=24 * 60)          # «граница ≤ 23:59»


def test_mutation_boundary_1900_is_caught():
    with pytest.raises(AssertionError):
        check_evening_invariant(cut_min=19 * 60)          # 18:55 попадает в день


def test_bar_1850_itself_excluded():
    b = synth(False)
    b.loc[len(b)] = [pd.Timestamp("2020-07-01 18:50"), 1, 999, 0.1, 1, 7]
    assert D.aggregate(b, "X").iloc[0]["close"] == 101.8


def _need():
    if not (CANDLES / "SBER.parquet").exists() or not OLD.exists():
        pytest.skip("нет общих данных")


def _old_dev(ticker):
    t = pq.read_table(OLD, filters=[("date", "<=", DEV_END.to_pydatetime()), ("ticker", "=", ticker)])
    return t.to_pandas().set_index("date").sort_index()


@pytest.mark.parametrize("ticker", ["SBER", "LKOH", "LSRG"])
def test_before_evening_session_equals_old_series(ticker):
    _need()
    new = D.aggregate(D.read_bars(CANDLES / f"{ticker}.parquet"), ticker).set_index("date")
    old = _old_dev(ticker)
    lim = pd.Timestamp("2020-06-20")
    n, o = new[new.index < lim], old[old.index < lim]
    assert len(n) > 500 and n.index.equals(o.index)
    # 30.07.2014 — единственный день с барами 18:50/19:00 (разовая вечерняя торговля): отличие по замыслу
    odd = pd.Timestamp("2014-07-30")
    assert (n.loc[odd, "close"] != o.loc[odd, "close"]) or odd not in n.index
    n, o = n.drop(index=odd), o.drop(index=odd)
    for c in ("open", "high", "low", "close", "volume"):
        assert n[c].to_numpy().tobytes() == o[c].to_numpy().astype(n[c].dtype).tobytes(), c


@pytest.mark.parametrize("ticker", ["SBER", "LKOH", "LSRG"])
def test_after_evening_close_is_bar_1845(ticker):
    _need()
    bars = D.read_bars(CANDLES / f"{ticker}.parquet")
    new = D.aggregate(bars, ticker).set_index("date")
    # баров с началом 18:45 в данных нет: последний бар с началом ≤ 18:45 — это 18:40 (аукцион закрытия)
    hm = bars["timestamp"].dt.strftime("%H:%M")
    b = bars[hm <= "18:45"].copy()
    b["date"] = b["timestamp"].dt.normalize()
    b = b[b["date"] >= pd.Timestamp("2020-08-10")].groupby("date").tail(1)
    assert (b["timestamp"].dt.strftime("%H:%M") == "18:40").mean() > 0.95
    b = b.set_index("date")["close"]
    days = b.index[:20]
    assert len(days) == 20
    np.testing.assert_array_equal(new.loc[days, "close"].to_numpy(), b.loc[days].to_numpy())
    assert (new["close"].notna()).all()


def test_no_data_from_2022_in_new_series():
    _need()
    d = D.build(MAIN, ["SBER"])
    assert d["date"].max() <= DEV_END
