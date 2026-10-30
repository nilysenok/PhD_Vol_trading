"""Официальный календарь основных сессий MOEX ≥ 2025 (Ж-142 п. 5; csv — docs/issledovaniya/2026-10-30_kalendar_moex_2025_2026.csv).

Основные сессии (в т. ч. перенесённые рабочие субботы, напр. 01.11.2025) — в дневном ряду; выходные сессии
ДСВД (osnovnaya_sessiya = 0) и праздники — нет. Подстановок нет: сессия без баров только отмечается в отчёте."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CALENDAR_MOEX = Path(__file__).resolve().parents[1] / "docs" / "issledovaniya" / "2026-10-30_kalendar_moex_2025_2026.csv"


def moex_sessions(path: Path | None = None) -> frozenset:
    """Даты основных сессий фондового рынка MOEX (osnovnaya_sessiya == 1; ДСВД и праздники — нет)."""
    c = pd.read_csv(path or CALENDAR_MOEX, parse_dates=["date"])
    return frozenset(c.loc[c["osnovnaya_sessiya"] == 1, "date"])


def sessions_without_bars(daily: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Основные сессии календаря в [start; end) без баров в ряду: по тикерам (для отчёта; подстановки нет)."""
    ses = sorted(d for d in moex_sessions() if start <= d < end)
    return {t: [d for d in ses if d not in set(g["date"])] for t, g in daily.groupby("ticker")}
