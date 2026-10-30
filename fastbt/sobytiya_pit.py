"""Календарь событий point-in-time (Ж-070 п. 2, Д3): data/events.csv.

sobytiya_na(d, ticker) отдаёт событие, только если data_obyavleniya <= d. Тикер `*` — всем тикерам.
Даты >= 2022 — PermissionError (и на запрос, и на строки файла). digest() — для ключей кэша (Д5).
К sobytiya_e1 не подключено (итерация 7).
"""
from __future__ import annotations

import csv
import hashlib
from datetime import date, timedelta
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "data" / "events.csv"
END = date(2022, 1, 1)
DEV_END = date(2022, 1, 1)   # граница периодов кроме wf; END подменяют тесты (эмуляция снятого замка), DEV_END — нет


def digest() -> str:
    """Д5: SHA1 файла событий."""
    return hashlib.sha1(Path(PATH).read_bytes()).hexdigest()


def _granitsa(period=None):
    """Граница чтения: период wf — 2026-03-01 (искл.) только при отметке снятого замка (period_v2.proverit_chtenie);
    без отметки — ZamokVladeltsa (PermissionError). Прочие периоды — END."""
    if period != "wf":
        return END
    from fastbt import period_v2
    period_v2.proverit_chtenie("wf")
    return period_v2.PERIODY["wf"].konec.date() + timedelta(days=1)


def _chitat(path=None) -> list:
    out = []
    with open(path or PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["data_obyavleniya"] = date.fromisoformat(r["data_obyavleniya"])
            r["data_sobytiya"] = date.fromisoformat(r["data_sobytiya"])
            out.append(r)
    return out


def _rows(path=None, end=None) -> list:
    """Строгое чтение: строка с датой >= end — PermissionError (период wf со снятым замком)."""
    end = end or END
    out = _chitat(path)
    for r in out:
        if r["data_sobytiya"] >= end or r["data_obyavleniya"] >= end:
            raise PermissionError(f"событие {r['tip']} {r['data_sobytiya']}: данные >= 2022 не читаются")
    return out


def _rows_dev(path=None) -> list:
    """Периоды кроме wf: файл содержит 2022+ ради wf; строки с датой >= DEV_END здесь не видны (граница периода,
    как до добавления данных). Запрос даты >= END отказывает в sobytiya_na."""
    return [r for r in _chitat(path) if r["data_sobytiya"] < DEV_END and r["data_obyavleniya"] < DEV_END]


def _vidimye(period, end, path=None) -> list:
    if period == "wf":
        return _rows(path, end)
    return _rows_dev(path) if END == DEV_END else _rows(path)


def sobytiya_na(d, ticker=None, vpered: int = 0, path=None, period=None) -> list:
    """События с data_sobytiya в [d, d+vpered] дней, известные к d (data_obyavleniya <= d).

    ticker=None — все; иначе события тикера и `*`. vpered=0 — события ровно дня d. period="wf" — см. _granitsa."""
    d = date.fromisoformat(str(d)[:10])
    end = _granitsa(period)
    if d >= end:
        raise PermissionError(f"{d}: данные >= {end} не читаются")
    konec = d + timedelta(days=vpered)
    return [r for r in _vidimye(period, end, path)
            if r["data_obyavleniya"] <= d
            and d <= r["data_sobytiya"] <= konec
            and (ticker is None or r["ticker"] in ("*", ticker))]
