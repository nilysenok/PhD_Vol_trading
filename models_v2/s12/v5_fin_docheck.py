"""V5 лист FIN, G24: сверка итогового документа (ТОЛЬКО ЧТЕНИЕ). Документ на диске = пересборка из таблиц на диске, число за числом: десятичные числа сравниваются попарно (сверено N, расхождений M), весь текст — побайтно.
Статические части не содержат результатных чисел: `static_decimals` перебирает таблицы с искажением и оставляет числа, не зависящие от них; они обязаны входить в `STATIC_DECIMALS` или быть нулями."""
from __future__ import annotations

import copy
import re

import numpy as np
import pandas as pd

from models_v2.s12 import v5_fin_asm as A, v5_fin_build as B, v5_fin_doc as W, v5_fin_text as TX

DEC = re.compile(r"\d+,\d+")
PARTS = ["Особый вопрос", "Реестр всех испытаний", "Отступления листов", "Причины отклонения", "Совместные ошибки", "Стоимость вычислений", "Абляция", "Статистическая значимость", "Полные формулы", "Математическое отличие от V3/V4",
         "QLIKE по всем 17 бумагам", "SBER и ориентир", "Результаты по годам", "Разбор критерия выбора", "Стабильность"]


def tokens(txt: str) -> list:
    return DEC.findall(txt)


def compare(doc: str, fresh: str) -> tuple:
    """-> (сверено чисел, расхождений): попарно по порядку в тексте; разная длина — лишние числа считаются расхождениями."""
    a, b = tokens(doc), tokens(fresh)
    return len(b), sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))


def check_doc(t: dict | None = None, doc: str | None = None) -> list:
    t = W.load() if t is None else t
    fresh = A.build(t)
    if doc is None and not B.FINAL.exists():
        return ["G24: нет файла %s" % B.FINAL.name]
    doc = B.FINAL.read_text(encoding="utf-8") if doc is None else doc
    n, bad = compare(doc, fresh)
    out = []
    if bad or doc != fresh:
        out.append("G24: документ ≠ пересборке из таблиц (чисел %d, расхождений %d)" % (n, bad))
    if doc.splitlines()[0] != W.first_line(t) or not doc.splitlines()[0].startswith("%d из %d конфигураций прошли «go»" % (W.n_go(t), 9)) or "исследовательский" not in doc[:2500]:
        out.append("G24: первая строка / статус")
    out += ["G24: нет раздела «%s»" % p for p in PARTS if p not in doc]
    if "bps" in doc.lower():
        out.append("G24: bps в документе (владельцу — проценты)")
    return out


def perturb(t: dict) -> dict:
    """Копия таблиц: каждое вещественное число ×1,31 + 0,1371, ранги EqY сдвигаются по кругу (год, счётчики и логические столбцы не трогаются)."""
    u = {}
    for k, v in t.items():
        if isinstance(v, pd.DataFrame):
            v = v.copy()
            for c in v.columns:
                if c == "rank_EqY":
                    v[c] = np.roll(v[c].to_numpy(), 7)
                elif pd.api.types.is_float_dtype(v[c]) and c not in ("year", "rc", "n_params", "n_dates", "n", "n_rows", "n_tickers", "n_rows_m16"):
                    v[c] = v[c] * 1.31 + 0.1371
        elif isinstance(v, dict):
            v = copy.deepcopy(v)
        u[k] = v
    return u


def static_decimals(t: dict) -> set:
    """Десятичные числа, не изменившиеся при искажении таблиц (т. е. не из таблиц), без нулевых."""
    orig = B.claims_fail
    B.claims_fail = lambda _t: []
    try:
        a, b = tokens(A.build(t)), tokens(A.build(perturb(t)))
    finally:
        B.claims_fail = orig
    return {x for x, y in zip(a, b) if x == y and float(x.replace(",", ".")) != 0.0} if len(a) == len(b) else set(a)


def check_static(t: dict) -> list:
    extra = static_decimals(t) - TX.STATIC_DECIMALS
    return ["G24: числа вне таблиц и вне списка определений: %s" % sorted(extra)] if extra else []


def count_numbers(t: dict) -> int:
    return len(tokens(A.build(t)))
