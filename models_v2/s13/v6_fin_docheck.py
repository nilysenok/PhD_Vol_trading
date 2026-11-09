"""V6 лист FIN, G52: сверка итогового документа (ТОЛЬКО ЧТЕНИЕ). Документ на диске = пересборка из таблиц на диске, число за числом: десятичные числа сравниваются попарно (сверено N, расхождений M), весь текст — побайтно.
Статические части не содержат результатных чисел: `static_decimals` перебирает таблицы с искажением и оставляет числа, не зависящие от них; они обязаны входить в `STATIC_DECIMALS` или быть нулями."""
from __future__ import annotations

import copy
import re

import pandas as pd

from models_v2.s13 import v6_fin_asm as A, v6_fin_doc as W, v6_fin_text as TX

DEC = re.compile(r"\d+,\d+")
PARTS = ["Результат", "Что исследовано в V1–V5", "Чем V6 отличается", "Формулы", "Полная таблица конфигураций", "QLIKE SBER по годам", "Сравнение с HAR-J", "Абляции", "Стресс-проверки",
         "Разложение причин", "Реестр испытаний", "Решение о перспективах", "Отступления программы"]


def tokens(txt: str) -> list:
    return DEC.findall(txt)


def compare(doc: str, fresh: str) -> tuple:
    """-> (сверено чисел, расхождений): попарно по порядку в тексте; разная длина — лишние числа считаются расхождениями."""
    a, b = tokens(doc), tokens(fresh)
    return len(b), sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))


def check_doc(t: dict | None = None, doc: str | None = None) -> list:
    t = W.load() if t is None else t
    fresh = A.build(t)
    if doc is None and not A.FINAL.exists():
        return ["G52: нет файла %s" % A.FINAL.name]
    doc = A.FINAL.read_text(encoding="utf-8") if doc is None else doc
    n, bad = compare(doc, fresh)
    out = []
    if bad or doc != fresh:
        out.append("G52: документ ≠ пересборке из таблиц (чисел %d, расхождений %d)" % (n, bad))
    if doc.splitlines()[0] != W.first_line(t) or not doc.startswith("Нет:") or "не независимый OOS" not in doc[:3000]:
        out.append("G52: первая строка / статус")
    heads = [ln for ln in doc.splitlines() if ln.startswith("## ")]
    out += ["G52: нет раздела %d «%s»" % (i + 1, p) for i, p in enumerate(PARTS) if len(heads) <= i or p not in heads[i]]
    if "bps" in doc.lower():
        out.append("G52: bps в документе (владельцу — проценты)")
    return out


def shift_text(x: str) -> str:
    return DEC.sub(lambda mm: mm.group() + "1", x)


def perturb(t: dict, k1: float = 1.31, k0: float = 0.1371) -> dict:
    """Копия таблиц: каждое вещественное число ×1,31 + 0,1371 (годы, счётчики, ранги, логические столбцы не трогаются); числа текстов-источников получают лишнюю цифру."""
    skip = {"year", "n", "n_days", "n_dates", "n_rows", "df", "level", "cond", "years_neg", "years_n", "window_n", "n_pool", "rc"}
    u = {}
    for k, v in t.items():
        if isinstance(v, pd.DataFrame):
            v = v.copy()
            for c in v.columns:
                if pd.api.types.is_float_dtype(v[c]) and c not in skip:
                    v[c] = v[c] * k1 + k0
        elif isinstance(v, dict):
            v = copy.deepcopy(v)
            if k == "cp":
                v["0"]["noise"]["mde_95_pct_of_k1"] = v["0"]["noise"]["mde_95_pct_of_k1"] * k1 + k0
        elif isinstance(v, str):
            v = shift_text(v)
        u[k] = v
    return u


def static_decimals(t: dict) -> set:
    """Десятичные числа, не изменившиеся при искажении таблиц (т. е. не из таблиц), без нулевых."""
    orig = A.claims_fail
    A.claims_fail = lambda _t: []
    try:
        a, b, c = tokens(A.build(t)), tokens(A.build(perturb(t))), tokens(A.build(perturb(t, 0.77, 0.0913)))
    finally:
        A.claims_fail = orig
    return {x for x, y, z in zip(a, b, c) if x == y == z and float(x.replace(",", ".")) != 0.0} if len(a) == len(b) == len(c) else set(a)


def check_static(t: dict) -> list:
    extra = static_decimals(t) - TX.STATIC_DECIMALS
    return ["G52: числа вне таблиц и вне списка определений: %s" % sorted(extra)] if extra else []


def count_numbers(t: dict) -> int:
    return len(tokens(A.build(t)))
