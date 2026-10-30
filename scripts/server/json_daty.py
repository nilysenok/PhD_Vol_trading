"""Даты внутри JSON-справочника (список записей) для manifest_perenos и usech_2021 (Ж-131 п. 2; лист манифеста v6).

Датой считается ЛЮБОЕ строковое значение вида ГГГГ-ММ-ДД или ГГГГ-ММ (месяц — его первое число) на любой глубине записи.
Запись входит в усечённую копию, только если ВСЕ её даты <= границы (пакет ≤ 2021: ни одной даты 2022). Для wf (граница
> 2021-12-31, pit) у записи с `frsttrade` датой границы служит он один: спецификация контракта известна с первой торговли (PIT),
lsttrade после границы не отбрасывает контракт — иначе активный контракт на конце окна пропадает. Используется для `data/iss_ext/br_contracts.json` (календарь контрактов Brent: frsttrade, lsttrade, month).
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


def daty_zapisi(z, pit: bool = False) -> list[pd.Timestamp]:
    """Даты записи для границы. pit=False (граница 2021-12-31) — все даты записи: в пакет ≤ 2021 не попадает ни одна дата 2022.
    pit=True (wf, граница > 2021-12-31) — запись со спецификацией (`frsttrade`) известна с `frsttrade`: граница — только он;
    lsttrade/month — запланированные параметры контракта. Нет `frsttrade` — все даты записи."""
    if pit and isinstance(z, dict) and isinstance(z.get("frsttrade"), str) and DATA.match(z["frsttrade"]):
        return _daty(z["frsttrade"])
    return _daty(z)


def _zapisi(p) -> list:
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    if not isinstance(d, list):
        raise ValueError("JSON не список записей")
    return d


def max_data(p, pit: bool = False) -> pd.Timestamp | str:
    """Max дата по всем записям (pit — см. daty_zapisi); нет дат — 'NODATES' (читатель манифеста)."""
    m = [d for z in _zapisi(p) for d in daty_zapisi(z, pit)]
    return max(m) if m else "NODATES"


def usech(src: Path, dst: Path, granica=None) -> tuple[int, int, str | None]:
    """Копия с записями, где все даты <= granica (умолч. GRANICA = 2021-12-31): (оставлено, отброшено, max дата)."""
    g = pd.Timestamp(GRANICA if granica is None else granica)
    pit = g > GRANICA
    ok = [z for z in _zapisi(src) if all(d <= g for d in daty_zapisi(z, pit))]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(ok, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    m = [d for z in ok for d in daty_zapisi(z, pit)]
    return len(ok), len(_zapisi(src)) - len(ok), str(max(m).date()) if m else None
