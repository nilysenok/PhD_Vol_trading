"""Даты внутри JSON-справочника (список записей) для manifest_perenos и usech_2021 (Ж-131 п. 2; лист манифеста v6).

Датой считается ЛЮБОЕ строковое значение вида ГГГГ-ММ-ДД или ГГГГ-ММ (месяц — его первое число) на любой глубине записи.
Запись входит в усечённую копию, только если ВСЕ её даты <= 2021-12-31 (как календари: событие в будущем не входит).
Используется для `data/iss_ext/br_contracts.json` (календарь контрактов Brent: frsttrade, lsttrade, month).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

DATA = re.compile(r"^(\d{4}-\d{2})(-\d{2})?$")
GRANICA = pd.Timestamp("2021-12-31")


def _daty(x) -> list[pd.Timestamp]:
    if isinstance(x, str):
        return [pd.Timestamp(x if len(x) == 10 else x + "-01")] if DATA.match(x) else []
    if isinstance(x, dict):
        x = list(x.values())
    return [d for v in x for d in _daty(v)] if isinstance(x, list) else []


def _zapisi(p) -> list:
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    if not isinstance(d, list):
        raise ValueError("JSON не список записей")
    return d


def max_data(p) -> pd.Timestamp | str:
    """Max дата по всем записям; нет дат — 'NODATES' (читатель манифеста)."""
    m = [d for z in _zapisi(p) for d in _daty(z)]
    return max(m) if m else "NODATES"


def usech(src: Path, dst: Path) -> tuple[int, int, str | None]:
    """Копия с записями, где все даты <= 2021-12-31: (оставлено, отброшено, max дата <= 2021)."""
    ok = [z for z in _zapisi(src) if all(d <= GRANICA for d in _daty(z))]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(ok, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    m = [d for z in ok for d in _daty(z)]
    return len(ok), len(_zapisi(src)) - len(ok), str(max(m).date()) if m else None
