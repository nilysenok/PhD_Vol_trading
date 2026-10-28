"""Агрегатор баров (Ж-098): синтетика, без сети и без данных проекта."""
import pandas as pd

from fastbt import bars_agg as ba


def _b(day: str, start: str, n: int, p0: float = 100.0) -> pd.DataFrame:
    ts = pd.date_range(f"{day} {start}", periods=n, freq="10min")
    i = pd.Series(range(n), dtype=float)
    return pd.DataFrame({"timestamp": ts, "open": p0 + i, "high": p0 + i + 0.5, "low": p0 + i - 0.5,
                         "close": p0 + i + 0.2, "volume": 10 + i})


def test_10m_v_1h_sohranyaet_ohlcv():
    b10 = _b("2016-01-04", "10:00", 6)
    r = ba.resample(ba.prep_10m(b10), ba.K["1h"]).iloc[0]
    assert r["open"] == b10["open"].iloc[0] and r["close"] == b10["close"].iloc[-1]
    assert r["high"] == b10["high"].max() and r["low"] == b10["low"].min() and r["volume"] == b10["volume"].sum()


def test_bary_posle_1850_i_predotkrytie_ne_uchastvuyut():
    b10 = pd.concat([_b("2016-01-04", "09:50", 1), _b("2016-01-04", "18:30", 4, p0=200.0)], ignore_index=True)
    p = ba.prep_10m(b10)  # 09:50 (предоткрытие), 18:50 и 19:00 отброшены; остались 18:30 и 18:40
    assert list(p["slot"]) == [ba.SLOTS - 2, ba.SLOTS - 1]
    d = ba.daily(p)
    assert d["volume"].iloc[0] == b10.loc[b10["timestamp"].isin(pd.to_datetime(["2016-01-04 18:30", "2016-01-04 18:40"])), "volume"].sum()


def test_nedelya_iz_dnej():
    days = ["2016-01-04", "2016-01-05", "2016-01-06", "2016-01-11"]  # три дня недели 1 и один — недели 2
    d = ba.daily(ba.prep_10m(pd.concat([_b(x, "10:00", 3, 100.0 + 10 * k) for k, x in enumerate(days)], ignore_index=True)))
    w = ba.weekly(d)
    assert len(w) == 2 and w["date"].iloc[0] == pd.Timestamp("2016-01-06")
    assert w["open"].iloc[0] == d["open"].iloc[0] and w["close"].iloc[0] == d["close"].iloc[2]
    assert w["high"].iloc[0] == d["high"].iloc[:3].max() and w["low"].iloc[0] == d["low"].iloc[:3].min()
    assert w["volume"].iloc[0] == d["volume"].iloc[:3].sum()
