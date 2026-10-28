"""Ж-097: признаки на синтетике — f2 = 1 при close = high, f3 = 0 без скачков, бары после 18:50 не видны."""
import numpy as np
import pandas as pd

from fastbt.intraday_priznaki import cell_stats, features_ticker, with_pit


def _bars(close, extra=None):
    ts = pd.date_range("2018-03-05 09:50", "2018-03-05 18:40", freq="10min")
    c = np.asarray(close, float)[: len(ts)]
    df = pd.DataFrame({"timestamp": ts, "open": c, "high": c + 0.01, "low": c - 0.01, "close": c, "volume": 100})
    if extra is not None:
        df = pd.concat([df, extra], ignore_index=True)
    return df


def _walk():
    n = len(pd.date_range("2018-03-05 09:50", "2018-03-05 18:40", freq="10min"))
    return 100.0 * np.cumprod(1 + 0.001 * np.where(np.arange(n) % 2, 1, -1))   # равные по модулю шаги, без скачков


def test_f2_close_equals_high():
    n = len(_walk())
    c = np.linspace(100, 101, n)                                  # растёт: close = максимум
    b = _bars(c)
    b["high"], b["low"] = b["close"], b["close"] - 0.5
    assert abs(features_ticker(b)["f2"].iloc[0] - 1.0) < 1e-12


def test_f3_zero_without_jumps():
    f = features_ticker(_bars(_walk()))
    assert f["f3"].iloc[0] == 0.0 and 0 < f["f1"].iloc[0] < 1


def test_features_do_not_see_bars_after_1850():
    base = _bars(_walk())
    ts = pd.date_range("2018-03-05 18:50", periods=6, freq="10min")
    ev = pd.DataFrame({"timestamp": ts, "open": 500., "high": 900., "low": 1., "close": 900., "volume": 10**9})
    a, b = features_ticker(base), features_ticker(_bars(_walk(), ev))
    pd.testing.assert_frame_equal(a, b)


def test_with_pit_replaces_f1_f3_only():
    f = features_ticker(_bars(_walk()))
    pit = pd.DataFrame({"f1": [0.25], "f3": [0.5]}, index=pd.MultiIndex.from_tuples([("X", f.index[0])], names=["ticker", "date"]))
    o = with_pit({"X": f}, pit)["X"]
    assert o["f1"].iloc[0] == 0.25 and o["f3"].iloc[0] == 0.5 and o["f2"].iloc[0] == f["f2"].iloc[0]


def test_cell_stats_spread_sign():
    rng = np.random.default_rng(0)
    cell = np.tile([0, 1, 2], 300)
    day = np.repeat(np.arange(300), 3)
    y = 0.01 * (cell - 1) + rng.normal(0, 0.001, 900)
    s = cell_stats(y, cell, day)
    assert s["spread"] > 0.015 and s["t"] > 10
