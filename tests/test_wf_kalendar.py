"""wf-kalendar (Ж-142 п. 5): дневка и календарь ≥ 2025 — по официальному календарю MOEX основных сессий.
Рабочая суббота 01.11.2025 — в ряду и календаре; выходные сессии ДСВД и праздники — нет; ≤ 2024 без изменений."""
from __future__ import annotations

import logging

import pandas as pd
import pytest

from fastbt import daily_1850 as D
from fastbt.kalendar_moex import sessions_without_bars
from test_wf_daily import bars, frame

MAIN = D.default_main()
CANDLES = MAIN / "moex_discovery" / "data" / "raw" / "candles_10m"
# пт 31.10, сб 01.11 (рабочая), пн 03.11 (торги), вт 04.11 (праздник), сб 08.11 и вс 09.11 (ДСВД)
DAYS = ["2025-10-31", "2025-11-01", "2025-11-03", "2025-11-04", "2025-11-08", "2025-11-09"]
SYN = frame(sum((bars(d) for d in DAYS), []))
WANT = ["2025-10-31", "2025-11-01", "2025-11-03"]


def got(agg) -> list:
    return [str(d.date()) for d in agg(SYN, "X")["date"]]


def check_series(agg=D.aggregate):
    assert got(agg) == WANT, got(agg)


def old_rule(bars_: pd.DataFrame, ticker: str, *a, **k) -> pd.DataFrame:
    """Мутация «вернуть WEEKEND_FROM»: все сб/вс ≥ 2025 выброшены, будни — как есть (прежнее правило #735)."""
    ts = bars_["timestamp"]
    kept = bars_[~((ts >= D.WEEKEND_FROM) & (ts.dt.dayofweek >= 5)) | (ts < D.WEEKEND_FROM)]
    return D.aggregate(kept, ticker, sessions=frozenset(pd.to_datetime(DAYS)))


def with_dsvd(bars_: pd.DataFrame, ticker: str, *a, **k) -> pd.DataFrame:
    """Мутация «включить ДСВД»: в календаре основных сессий оказались все дни csv, и выходных сессий тоже."""
    c = pd.read_csv(D.CALENDAR_MOEX, parse_dates=["date"])
    return D.aggregate(bars_, ticker, sessions=frozenset(c.loc[c["is_traded"] == 1, "date"]))


def test_workday_saturday_2025_11_01_in_series():
    check_series()


def test_dsvd_saturday_and_holiday_not_in_series():
    g = got(D.aggregate)
    assert "2025-11-08" not in g and "2025-11-09" not in g and "2025-11-04" not in g


def test_mutation_old_weekend_rule_caught():
    with pytest.raises(AssertionError):
        check_series(old_rule)


def test_mutation_dsvd_included_caught():
    with pytest.raises(AssertionError):
        check_series(with_dsvd)


def test_mutation_weekend_from_restored_caught(monkeypatch):
    """WEEKEND_FROM далеко в будущем = правило календаря выключено, DSVD попадает в ряд."""
    monkeypatch.setattr(D, "WEEKEND_FROM", pd.Timestamp("2100-01-01"))
    with pytest.raises(AssertionError):
        check_series()


def test_sessions_only_main():
    s = D.moex_sessions()
    assert pd.Timestamp("2025-11-01") in s
    assert pd.Timestamp("2025-11-08") not in s and pd.Timestamp("2026-02-23") not in s     # ДСВД, в т. ч. в праздник
    assert pd.Timestamp("2025-01-01") not in s and len(s) == 292


def test_bars_without_session_logged_not_included(caplog):
    with caplog.at_level(logging.INFO, logger=D.log.name):
        D.aggregate(SYN, "X")
    assert "X" in caplog.text and "3" in caplog.text           # 04.11, 08.11, 09.11


def test_calendar_extend_includes_workday_saturday():
    daily = pd.DataFrame({"ticker": "X", "date": pd.to_datetime(DAYS)})
    new = D.calendar_extend(["2025-10-30"], daily)
    assert [str(d.date()) for d in new] == ["2025-10-30"] + WANT


def test_mutation_calendar_old_rule_caught(monkeypatch):
    monkeypatch.setattr(D, "moex_sessions", lambda *a: frozenset(pd.to_datetime(DAYS)))
    daily = pd.DataFrame({"ticker": "X", "date": pd.to_datetime(DAYS)})
    with pytest.raises(AssertionError):
        assert [str(d.date()) for d in D.calendar_extend(["2025-10-30"], daily)] == ["2025-10-30"] + WANT


def test_up_to_2024_unchanged_weekends_kept():
    """До 2025 правило прежнее: рабочие субботы 2024 и любые дни с барами остаются, праздник 01.01.2024 тоже."""
    ds = ["2024-01-01", "2024-04-27", "2024-11-02", "2024-12-28"]
    assert [str(d.date()) for d in D.aggregate(frame(sum((bars(d) for d in ds), [])), "X")["date"]] == ds


def test_session_without_bars_reported_not_filled():
    daily = pd.DataFrame({"ticker": "X", "date": pd.to_datetime(["2025-01-03"])})
    miss = sessions_without_bars(daily, D.WEEKEND_FROM, pd.Timestamp("2025-01-10"))["X"]
    assert [str(d.date()) for d in miss] == ["2025-01-06", "2025-01-08", "2025-01-09"] and len(daily) == 1


@pytest.mark.skipif(not (CANDLES / "SBER.parquet").exists() or not D.out_path().exists(), reason="нет данных")
def test_real_up_to_2021_same_and_2024_unchanged_and_saturday_in():
    cur = pd.read_parquet(D.out_path())
    cur = cur[cur["ticker"] == "SBER"].reset_index(drop=True)
    wf = D.build_wf(tickers=["SBER"])
    pd.testing.assert_frame_equal(cur, wf[wf["date"] < D.END].reset_index(drop=True))
    wd = wf["date"]
    assert pd.Timestamp("2025-11-01") in set(wd)
    assert not ((wd >= "2025-11-02") & (wd.dt.dayofweek >= 5)).any()
    old = D.build_wf(tickers=["SBER"], end=pd.Timestamp("2025-01-01"))
    pd.testing.assert_frame_equal(old, wf[wf["date"] < "2025-01-01"].reset_index(drop=True))
