"""Граница чтения данных (реверс раунд 1, Ж-117): зона dev (< 2022-01-01, умолчание) и зона Z2 «тест-1» (< 2025-01-01).

`end()` — текущая граница, исключительно; загрузчики спрашивают её при вызове, а не при импорте.
`HARD` — граница навсегда (hold-out ≥ 2025): любое чтение за ней — PermissionError в любой зоне.
Пути файлов Z2 (после сужения замка) — переменные окружения FASTBT_Z2_<ВИД>: DAILY (ряд 18:50, parquet), DIV (таблица
дивидендов 2022–2024, csv), IMOEX (parquet 10-минуток), VPRED (σ̂ для TEST_C/D); CBR — data/calendar/cbr.csv как есть.
Нужного пути нет — понятная ошибка, а не тихое чтение файла зоны dev.
Зона `wf` (Ж-136, walk-forward): граница чтения — 2026-03-01 исключительно (последний день 2026-02-28); включается
только при отметке снятого замка (`period_v2.proverit_chtenie`, `docs/resheniya/wf_zamok_snyat.json`), иначе ZamokVladeltsa.
В зоне wf HARD заменяется на END_WF; в прочих зонах — прежний HARD (2025-01-01). Файлы wf лежат по путям dev (path() — default).
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path

import pandas as pd

END_DEV = pd.Timestamp("2022-01-01")
END_Z2 = pd.Timestamp("2025-01-01")
END_WF = pd.Timestamp("2026-03-01")   # исключительно: читается по 2026-02-28 включительно
HARD = END_Z2
_ENDS = {"dev": END_DEV, "test1": END_Z2, "wf": END_WF}
_zone = "dev"
_hooks: list = []


def on_change(fn) -> None:
    """fn вызывается при каждой смене зоны (сброс кэшей загрузчиков)."""
    _hooks.append(fn)


def name() -> str:
    return _zone


def end() -> pd.Timestamp:
    return _ENDS[_zone]


def hard() -> pd.Timestamp:
    """Граница навсегда для текущей зоны: wf — END_WF (замок снят отметкой), иначе HARD."""
    return END_WF if _zone == "wf" else HARD


def check(d, what: str = "данные") -> None:
    """d (дата или максимум ряда) должна быть строго меньше hard(); граница текущей зоны проверяется загрузчиком через end()."""
    h = hard()
    if pd.Timestamp(d) >= h:
        raise PermissionError(f"{what}: даты ≥ {h.date()} закрыты до hold-out")


def path(kind: str, default: Path | None = None) -> Path | None:
    """Файл вида kind: в зоне dev — default; в Z2 — FASTBT_Z2_<KIND> (нет переменной — FileNotFoundError)."""
    if _zone in ("dev", "wf"):
        return default
    p = os.environ.get(f"FASTBT_Z2_{kind.upper()}")
    if not p:
        raise FileNotFoundError(f"зона {_zone}: не задан путь FASTBT_Z2_{kind.upper()}")
    return Path(p)


NET = "NET"                      # FASTBT_Z2_DIV=NET — строка-маркер, не путь: тест-1 без дивидендов (Ж-117, вариант б)
SHAPKA_DIV = "# дивиденды: НЕТ (2022–2024 без дивидендов, Ж-117, вариант б)\n"


def div_net() -> bool:
    """True — режим «без дивидендов 2022–2024». Маркер NET допустим только в зоне test1; в dev — ValueError."""
    if os.environ.get("FASTBT_Z2_DIV") != NET:
        return False
    if _zone != "test1":
        raise ValueError(f"FASTBT_Z2_DIV=NET допустим только в зоне test1, сейчас {_zone!r}")
    return True


def shapka_dividendy() -> str:
    """Строка шапки csv реверса: «дивиденды: НЕТ», если задан NET; иначе пусто. Зону не проверяет (отказ — в div_net())."""
    return SHAPKA_DIV if os.environ.get("FASTBT_Z2_DIV") == NET else ""


def set_zone(zone: str) -> None:
    global _zone
    if zone not in _ENDS:
        raise ValueError(f"зона {zone!r}: есть {sorted(_ENDS)}")
    if zone == _zone:
        return
    if zone == "wf":
        from fastbt import period_v2
        period_v2.proverit_chtenie("wf", otmetka=period_v2.OTMETKA)   # нет отметки снятого замка — ZamokVladeltsa
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
