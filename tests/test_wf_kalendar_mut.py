"""wf-kalendar, тестировщик #754: мутации (г) и граница — календарь не найден не должен молча давать старое правило."""
from __future__ import annotations

import pandas as pd
import pytest

from fastbt import daily_1850 as D
from fastbt import kalendar_moex as K
from tests.test_wf_kalendar import DAYS, SYN


def test_missing_calendar_csv_refuses(tmp_path):
    """Нет csv — отказ (FileNotFoundError), а не пустой/старый календарь."""
    with pytest.raises(FileNotFoundError):
        K.moex_sessions(tmp_path / "net.csv")


def test_missing_calendar_in_aggregate_refuses(monkeypatch, tmp_path):
    monkeypatch.setattr(K, "CALENDAR_MOEX", tmp_path / "net.csv")
    monkeypatch.setattr(D, "moex_sessions", lambda *a: K.moex_sessions(tmp_path / "net.csv"))
    with pytest.raises(FileNotFoundError):
        D.aggregate(SYN, "X")


def test_mutation_silent_fallback_caught(monkeypatch):
    """Мутация (г): при ненайденном csv — молча «все будни» (старое правило). Такая подмена даёт 04.11 (праздник) в ряду."""
    def fallback(*a):
        return frozenset(d for d in pd.to_datetime(DAYS) if d.dayofweek < 5)
    monkeypatch.setattr(D, "moex_sessions", fallback)
    got = [str(d.date()) for d in D.aggregate(SYN, "X")["date"]]
    assert got != ["2025-10-31", "2025-11-01", "2025-11-03"]


def test_calendar_csv_has_no_gaps_in_2025_2026_weekdays():
    """Календарь: у каждой даты csv ровно одна строка; 01.11.2025 основная сессия, 08.11.2025 — нет."""
    c = pd.read_csv(K.CALENDAR_MOEX, parse_dates=["date"])
    assert c["date"].is_unique
    s = K.moex_sessions()
    assert pd.Timestamp("2025-11-01") in s and pd.Timestamp("2025-11-08") not in s
