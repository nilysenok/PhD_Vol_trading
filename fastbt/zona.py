"""Граница чтения данных (реверс раунд 1, Ж-117): зона dev (< 2022-01-01, умолчание) и зона Z2 «тест-1» (< 2025-01-01).

`end()` — текущая граница, исключительно; загрузчики спрашивают её при вызове, а не при импорте.
`HARD` — граница навсегда (hold-out ≥ 2025): любое чтение за ней — PermissionError в любой зоне.
Пути файлов Z2 (после сужения замка) — переменные окружения FASTBT_Z2_<ВИД>: DAILY (ряд 18:50, parquet), DIV (таблица
дивидендов 2022–2024, csv), IMOEX (parquet 10-минуток), VPRED (σ̂ для TEST_C/D); CBR — data/calendar/cbr.csv как есть.
Нужного пути нет — понятная ошибка, а не тихое чтение файла зоны dev.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path

import pandas as pd

END_DEV = pd.Timestamp("2022-01-01")
END_Z2 = pd.Timestamp("2025-01-01")
HARD = END_Z2
_ENDS = {"dev": END_DEV, "test1": END_Z2}
_zone = "dev"
_hooks: list = []


def on_change(fn) -> None:
    """fn вызывается при каждой смене зоны (сброс кэшей загрузчиков)."""
    _hooks.append(fn)


def name() -> str:
    return _zone


def end() -> pd.Timestamp:
    return _ENDS[_zone]


def check(d, what: str = "данные") -> None:
    """d (дата или максимум ряда) должна быть строго меньше HARD; граница текущей зоны проверяется загрузчиком через end()."""
    if pd.Timestamp(d) >= HARD:
        raise PermissionError(f"{what}: даты ≥ {HARD.date()} закрыты до hold-out")


def path(kind: str, default: Path | None = None) -> Path | None:
    """Файл вида kind: в зоне dev — default; в Z2 — FASTBT_Z2_<KIND> (нет переменной — FileNotFoundError)."""
    if _zone == "dev":
        return default
    p = os.environ.get(f"FASTBT_Z2_{kind.upper()}")
    if not p:
        raise FileNotFoundError(f"зона {_zone}: не задан путь FASTBT_Z2_{kind.upper()}")
    return Path(p)


def set_zone(zone: str) -> None:
    global _zone
    if zone not in _ENDS:
        raise ValueError(f"зона {zone!r}: есть {sorted(_ENDS)}")
    if zone == _zone:
        return
    _zone = zone
    for fn in _hooks:
        fn()


@contextlib.contextmanager
def zone_ctx(zone: str):
    old = _zone
    set_zone(zone)
    try:
        yield
    finally:
        set_zone(old)
