"""V7-1b-C: проверка реестра `ALL_TRIALS.csv` после дописывания строк результата. Строки регистрации (заголовок + 11 строк плана, записаны 09.11 16:44) не изменены побайтно (sha256 первых 12 строк);
всё, что ниже, — строки результата, ссылающиеся только на зарегистрированные конфигурации. Заменяет заглушку «ровно 12 строк» листа 1a (решение дирижёра [V7], класс B, 09.11)."""
from __future__ import annotations

import hashlib
from pathlib import Path

from models_v2.s14 import v7_1m_reg as RG

REG_LINES = 12
REG_SHA256 = "02b9be67861f840844c5a303d62a50bfa7db2f3698a4b8a87caeaaadbdf1190f"


def registration_intact(path: Path | None = None) -> bool:
    lines = (path or RG.TRIALS).read_bytes().splitlines(keepends=True)
    return len(lines) >= REG_LINES and hashlib.sha256(b"".join(lines[:REG_LINES])).hexdigest() == REG_SHA256


def results_only_registered(path: Path | None = None) -> bool:
    """Строки после регистрации — результаты (kind = result, статус «посчитана») зарегистрированных конфигураций, без дублей."""
    rows = RG.read(path)[REG_LINES:]
    decl = {"Ж-233:%s" % i: None for i in RG.IDS}
    keys = [r[0] for r in rows]
    return len(keys) == len(set(keys)) and all(RG.is_result(r, decl) for r in rows)
