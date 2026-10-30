"""Тестировщик PR #735: мутации, не пойманные tests/test_wf_daily.py (границы WEEKEND_FROM, замок build_wf,
END_WF, бар 18:50, равенство build_wf и build на ≤ 2021). Синтетика в tmp_path, сеть и реальные данные не нужны."""
from __future__ import annotations

import pandas as pd
import pytest

from fastbt import daily_1850 as D
from fastbt import period_v2
from test_wf_daily import bars, frame


def days(agg_df) -> list:
    return [str(d.date()) for d in agg_df["date"]]


def test_workday_saturdays_2024_kept():
    """Рабочие субботы-переносы 2024 (27.04, 02.11, 28.12) остаются; мутация WEEKEND_FROM=2024-01-01 их выбрасывает."""
    ds = ["2024-04-27", "2024-11-02", "2024-12-28"]
    rows = sum((bars(d) for d in ds), [])
    assert days(D.aggregate(frame(rows), "X")) == ds


def test_weekend_boundary_2025():
    """Константа ровно 2025-01-01. 2024-12-31 вт остаётся (до границы); 2025-01-01 — праздник MOEX, нет;
    2025-01-03 пт и 01-06 пн — основные сессии, остаются; 2025-01-04 сб и 01-05 вс — нет."""
    assert D.WEEKEND_FROM == pd.Timestamp("2025-01-01")
    rows = sum((bars(d) for d in ["2024-12-31", "2025-01-01", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06"]), [])
    assert days(D.aggregate(frame(rows), "X")) == ["2024-12-31", "2025-01-03", "2025-01-06"]


def test_calendar_workday_saturday_2024_kept():
    old = ["2024-04-26"]
    new = D.calendar_extend(old, pd.DataFrame({"ticker": "X", "date": pd.to_datetime(
        ["2024-04-27", "2024-04-29", "2025-01-04", "2025-01-06"])}))
    assert [str(d.date()) for d in new] == ["2024-04-26", "2024-04-27", "2024-04-29", "2025-01-06"]


def test_bar_1850_not_in_close():
    """Бар с началом ровно 18:50 в день не входит: close = close бара 18:40 (так в aggregate: mod < 18:50)."""
    rows = bars("2025-03-10") + [("2025-03-10 18:50", 101.5, 200, 10, 55.0, 999)]
    r = D.aggregate(frame(rows), "X").iloc[0]
    assert r["close"] == 101.5 and r["high"] == 102 and r["low"] == 99 and r["volume"] == 40


def test_bar_1840_is_in_day():
    rows = bars("2025-03-10")
    assert D.aggregate(frame(rows), "X").iloc[0]["close"] == 101.5


# ---------- build_wf на синтетических 10-минутках ----------
@pytest.fixture
def main(tmp_path):
    d = tmp_path / "moex_discovery" / "data" / "raw" / "candles_10m"
    d.mkdir(parents=True)
    ds = ["2019-06-01", "2021-12-31", "2026-02-27", "2026-03-02"]
    rows = sum((bars(x) for x in ds), [])
    frame(rows).to_parquet(d / "X.parquet", index=False)
    return tmp_path


def test_build_wf_requires_mark(main, monkeypatch):
    """Без отметки владельца build_wf не читает: проверка вызвана с периодом 'wf' и её отказ доходит наружу."""
    calls = []

    def deny(period, *a, **k):
        calls.append(period)
        raise period_v2.ZamokVladeltsa("нет отметки")
    monkeypatch.setattr(period_v2, "proverit_chtenie", deny)
    with pytest.raises(period_v2.ZamokVladeltsa):
        D.build_wf(main, ["X"])
    assert calls == ["wf"]


def test_build_wf_end_and_prefix(main, monkeypatch):
    monkeypatch.setattr(period_v2, "proverit_chtenie", lambda *a, **k: None)
    assert D.END_WF == pd.Timestamp("2026-03-01")
    wf = D.build_wf(main, ["X"])
    assert days(wf) == ["2019-06-01", "2021-12-31", "2026-02-27"]        # 2026-03-02 за границей
    old = D.build(main, ["X"])
    assert days(old) == ["2019-06-01", "2021-12-31"]
    pd.testing.assert_frame_equal(old, wf[wf["date"] < D.END].reset_index(drop=True))


def test_build_wf_no_weekend_since_2025(tmp_path, monkeypatch):
    monkeypatch.setattr(period_v2, "proverit_chtenie", lambda *a, **k: None)
    d = tmp_path / "moex_discovery" / "data" / "raw" / "candles_10m"
    d.mkdir(parents=True)
    ds = ["2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06"]
    frame(sum((bars(x) for x in ds), [])).to_parquet(d / "X.parquet", index=False)
    assert days(D.build_wf(tmp_path, ["X"])) == ["2025-01-03", "2025-01-06"]
