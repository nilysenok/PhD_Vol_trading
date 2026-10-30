"""Календарь экспираций MIX: MXH2 экспирировал 2022-03-29, а не в третий четверг 17.03 (Д-1, 29.10).

Источник: ISS MXH2 (LSTDELDATE) и https://www.moex.com/n45596. Мутации — только в памяти."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CSV = Path(__file__).resolve().parents[1] / "data" / "calendar" / "expirations.csv"
ISKLYUCHENIYA = {pd.Timestamp("2022-03-29")}


def _load() -> pd.DataFrame:
    d = pd.read_csv(CSV, parse_dates=["date", "known_from", "known_to"])
    return d[d["known_to"].isna()].reset_index(drop=True)  # действующие версии; история — test_calendar_expirations_pit.py


def _tretiy_chetverg(d: pd.Timestamp) -> bool:
    return d.dayofweek == 3 and 15 <= d.day <= 21


def _narusheniya(d: pd.DataFrame) -> list[str]:
    """Март 2022 обязан быть 29.03; прочие даты — третий четверг."""
    bad = []
    m22 = d[(d["date"].dt.year == 2022) & (d["date"].dt.month == 3)]
    if list(m22["date"]) != [pd.Timestamp("2022-03-29")]:
        bad.append(f"март 2022: {[str(x.date()) for x in m22['date']]}")
    for x in d["date"]:
        if x not in ISKLYUCHENIYA and not _tretiy_chetverg(x):
            bad.append(f"не третий четверг: {x.date()}")
    return bad


def test_mxh2_29_marta_2022() -> None:
    assert _narusheniya(_load()) == []


def test_ostalnye_kvartaly_ne_izmeneny() -> None:
    d = _load()
    assert len(d) == 52 and d["date"].is_monotonic_increasing and d["date"].is_unique
    assert len(d[d["date"].dt.year == 2022]) == 4


def test_razrabotka_ne_vidit_pravki() -> None:
    """Строка 2022 известна с 2022-01-01: на днях ≤ 2021 числа разработки не меняются."""
    d = _load()
    r = d[d["date"] >= "2022-01-01"]
    assert (r["known_from"] >= pd.Timestamp("2022-01-01")).all()


def test_mutatsiya_staraya_data_lovitsya() -> None:
    d = _load()
    d.loc[d["date"] == "2022-03-29", "date"] = pd.Timestamp("2022-03-17")
    assert _narusheniya(d) != []
