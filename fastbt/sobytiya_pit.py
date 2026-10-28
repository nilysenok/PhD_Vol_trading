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


def digest() -> str:
    """Д5: SHA1 файла событий."""
    return hashlib.sha1(Path(PATH).read_bytes()).hexdigest()


def _rows(path=None) -> list:
    out = []
    with open(path or PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["data_obyavleniya"] = date.fromisoformat(r["data_obyavleniya"])
            r["data_sobytiya"] = date.fromisoformat(r["data_sobytiya"])
            if r["data_sobytiya"] >= END or r["data_obyavleniya"] >= END:
                raise PermissionError(f"событие {r['tip']} {r['data_sobytiya']}: данные >= 2022 не читаются")
            out.append(r)
    return out


def sobytiya_na(d, ticker=None, vpered: int = 0, path=None) -> list:
    """События с data_sobytiya в [d, d+vpered] дней, известные к d (data_obyavleniya <= d).

    ticker=None — все; иначе события тикера и `*`. vpered=0 — события ровно дня d."""
    d = date.fromisoformat(str(d)[:10])
    if d >= END:
        raise PermissionError(f"{d}: данные >= 2022 не читаются")
    konec = d + timedelta(days=vpered)
    return [r for r in _rows(path)
            if r["data_obyavleniya"] <= d
            and d <= r["data_sobytiya"] <= konec
            and (ticker is None or r["ticker"] in ("*", ticker))]
