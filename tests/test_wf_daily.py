"""wf-L2: дневной ряд 18:50 и календарь до 2026-02-28; выходные сессии ≥ 2025 исключены; ≤ 2021 побайтно прежний."""
from __future__ import annotations

import pandas as pd
import pytest

from fastbt import daily_1850 as D

MAIN = D.default_main()
CANDLES = MAIN / "moex_discovery" / "data" / "raw" / "candles_10m"


def bars(day: str, close_after: float | None = None) -> list:
    rows = [(f"{day} 09:50", 100, 101, 99, 100.5, 10), (f"{day} 18:40", 100.5, 102, 100, 101.5, 30)]
    if close_after is not None:                       # бар после 18:50 с другой ценой
        rows.append((f"{day} 19:10", 101.5, 150, 50, close_after, 1000))
    return rows


def frame(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"]).assign(
        timestamp=lambda x: pd.to_datetime(x["timestamp"]))


# 2025-03-07 пт, 03-08 сб, 03-09 вс, 03-10 пн; 2019-06-01 суббота (до границы: остаётся)
SYN = frame(bars("2025-03-07") + bars("2025-03-08") + bars("2025-03-09") + bars("2025-03-10"))


def check_weekends_excluded(agg=D.aggregate):
    got = [str(d.date()) for d in agg(SYN, "X")["date"]]
    assert got == ["2025-03-07", "2025-03-10"], got


def test_weekend_session_not_in_daily():
    check_weekends_excluded()


def test_mutation_weekends_included_caught(monkeypatch):
    monkeypatch.setattr(D, "WEEKEND_FROM", pd.Timestamp("2100-01-01"))
    with pytest.raises(AssertionError):
        check_weekends_excluded()


def test_weekend_before_2025_kept():
    got = D.aggregate(frame(bars("2019-05-31") + bars("2019-06-01")), "X")
    assert [str(d.date()) for d in got["date"]] == ["2019-05-31", "2019-06-01"]


def check_close_1850():
    r = D.aggregate(frame(bars("2025-03-10", close_after=77.0)), "X").iloc[0]
    assert r["close"] == 101.5 and r["high"] == 102 and r["volume"] == 40


def test_close_is_last_bar_before_1850():
    check_close_1850()


def test_mutation_close_after_1850_caught():
    with pytest.raises(AssertionError):
        r = D.aggregate(frame(bars("2025-03-10", close_after=77.0)), "X", 24 * 60).iloc[0]   # граница 24:00
        assert r["close"] == 101.5 and r["volume"] == 40


def daily(days: list) -> pd.DataFrame:
    return pd.DataFrame({"ticker": "X", "date": pd.to_datetime(days)})


def check_calendar(ext=D.calendar_extend):
    old = ["2024-12-27", "2024-12-28", "2024-12-30"]
    new = ext(old, daily(["2024-12-30", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06"]))
    assert [str(d.date()) for d in new] == ["2024-12-27", "2024-12-28", "2024-12-30", "2025-01-03", "2025-01-06"]


def test_calendar_no_weekends_and_prefix_kept():
    check_calendar()


def test_mutation_calendar_with_weekends_caught(monkeypatch):
    monkeypatch.setattr(D, "WEEKEND_FROM", pd.Timestamp("2100-01-01"))
    with pytest.raises(AssertionError):
        check_calendar()


@pytest.mark.skipif(not (CANDLES / "SBER.parquet").exists() or not D.out_path().exists(), reason="нет данных")
@pytest.mark.parametrize("t", ["SBER", "LSRG", "OGKB"])
def test_up_to_2021_same_as_current_series(t):
    """Участок < 2022 из build_wf совпадает с текущим рядом (значения и dtype); файл не пишется."""
    cur = pd.read_parquet(D.out_path())
    cur = cur[cur["ticker"] == t].reset_index(drop=True)
    new = D.build_wf(tickers=[t])
    new = new[new["date"] < D.END].reset_index(drop=True)
    pd.testing.assert_frame_equal(cur, new)
    assert new["date"].max() < D.END
    wf = D.build_wf(tickers=[t])
    # нет сб/вс ≥ 2025, кроме основных сессий календаря MOEX (рабочая суббота 01.11.2025); ДСВД и праздников нет
    wk = wf[(wf["date"] >= D.WEEKEND_FROM) & (wf["date"].dt.dayofweek >= 5)]
    assert set(wk["date"]) <= D.moex_sessions()
    assert set(wf.loc[wf["date"] >= D.WEEKEND_FROM, "date"]) <= D.moex_sessions()
