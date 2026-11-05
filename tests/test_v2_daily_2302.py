"""↺ 05.11.2026: с 01.03.2025 исключаются дни, когда IMOEX не рассчитывался (нет баров основной сессии): 23.02.2026.

Мутация «правило только Сб/Вс» (убрать nonworking_2025) ловится реальным рядом и синтетикой.
"""
import numpy as np
import pandas as pd
import pytest

from models_v2 import daily, prov

A = 0.001


def mk_day(date, rets):
    t0 = pd.Timestamp(date) + pd.Timedelta(hours=10)
    p = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[100.0], p[:-1]])
    return pd.DataFrame({"begin": [t0 + pd.Timedelta(minutes=10 * k) for k in range(len(rets))],
                         "open": open_, "high": np.maximum(open_, p), "low": np.minimum(open_, p), "close": p})


def bars_of(days):
    return pd.concat([mk_day(d, A * np.tile([1.0, -1.0], 26)[:51]) for d in days])


def test_weekday_without_index_session_excluded_from_2025_03():
    b = bars_of(["2025-03-03", "2025-03-04", "2025-03-05"])
    d, ex = daily.build_daily(b, index_days={pd.Timestamp("2025-03-03"), pd.Timestamp("2025-03-05")})
    assert list(d.index) == [pd.Timestamp("2025-03-03"), pd.Timestamp("2025-03-05")]
    assert ex[["date", "reason"]].values.tolist() == [[pd.Timestamp("2025-03-04"), "nonworking_2025"]]


def test_rule_not_applied_before_2025_03():
    b = bars_of(["2025-02-26", "2025-02-27"])
    d, ex = daily.build_daily(b, index_days=set())
    assert len(d) == 2 and ex.empty


def test_rule_off_for_index_itself():
    d, ex = daily.build_daily(bars_of(["2025-03-03", "2025-03-04"]), index_days=None)
    assert len(d) == 2 and ex.empty


def test_real_series_has_no_23_feb_2026_and_exactly_81_excluded():
    bars = pd.read_parquet(prov.BARS)
    d, ex = daily.build_daily(bars)
    assert pd.Timestamp("2026-02-23") not in d.index
    row = ex[ex["date"] == pd.Timestamp("2026-02-23")]
    assert row["reason"].tolist() == ["nonworking_2025"] and row["n_bars_main"].tolist() == [53]
    assert len(ex) == 81 and ex["reason"].value_counts().to_dict() == {"weekend_2025": 75, "bars<30": 4, "stop_2022": 1, "nonworking_2025": 1}
    assert len(d) == 2944


def test_every_remaining_day_from_2025_03_has_index_session():
    bars = pd.read_parquet(prov.BARS)
    d, _ = daily.build_daily(bars)
    idays = daily.index_session_days()
    late = d.index[d.index >= daily.WEEKEND_FROM]
    assert [x for x in late if x not in idays] == []
