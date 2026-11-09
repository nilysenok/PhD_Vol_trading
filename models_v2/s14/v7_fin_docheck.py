"""V7 лист FIN, G50: сверка итогового документа (ТОЛЬКО ЧТЕНИЕ). Документ на диске = пересборка из таблиц на диске, число за числом: десятичные числа сравниваются попарно (сверено N, расхождений M), весь текст — побайтно.
Статические части не содержат результатных чисел: `static_decimals` перебирает таблицы с искажением и оставляет числа, не зависящие от них; они обязаны входить в `STATIC_DECIMALS` или быть нулями."""
from __future__ import annotations

import copy
import re

import pandas as pd

from models_v2.s14 import v7_fin_asm as A, v7_fin_doc as W, v7_fin_text as TX

DEC = re.compile(r"\d+,\d+")


def tokens(txt: str) -> list:
    return DEC.findall(txt)


def compare(doc: str, fresh: str) -> tuple:
    """-> (сверено чисел, расхождений): попарно по порядку; разная длина — лишние числа считаются расхождениями."""
    a, b = tokens(doc), tokens(fresh)
    return len(b), sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))


def check_doc(t: dict | None = None, doc: str | None = None) -> list:
    t = A.load() if t is None else t
    if doc is None and not A.FINAL.exists():
        return ["G50: нет файла %s" % A.FINAL.name]
    doc = A.FINAL.read_text(encoding="utf-8") if doc is None else doc
    fresh = A.build(t)
    n, bad = compare(doc, fresh)
    out = []
    if bad or doc != fresh:
        out.append("G50: документ ≠ пересборке из таблиц (чисел %d, расхождений %d)" % (n, bad))
    lines = doc.splitlines()
    if lines[0] != W.first_line(t) or lines[2] != TX.STATUS:
        out.append("G50: первая строка (результат) или шапка статуса")
    heads = [ln for ln in lines if ln.startswith("## ")]
    out += ["G50: нет раздела %d «%s»" % (i + 1, p) for i, p in enumerate(A.SECTIONS) if len(heads) <= i or p not in heads[i]]
    if "bps" in doc.lower():
        out.append("G50: bps в документе (владельцу — проценты)")
    return out


def shift_text(x: str) -> str:
    return DEC.sub(lambda mm: mm.group() + "1", x)


def perturb(t: dict, k1: float = 1.31, k0: float = 0.1371) -> dict:
    """Копия таблиц: каждое вещественное число ×1,31 + 0,1371 (годы, счётчики не трогаются); текстовые столбцы и числа в текстах получают лишнюю цифру."""
    skip = {"year", "n", "n_days", "n_bars", "n_a", "n_b", "line", "cond", "years_neg", "n_years", "place", "n_rows", "n_crisis", "n_2018_2021", "n_2022_2025", "k", "bars_mode", "days_not_mode", "cfg", "lam_g"}
    u = {}
    for k, v in t.items():
        if isinstance(v, pd.DataFrame):
            v = v.copy()
            for c in v.columns:
                if pd.api.types.is_float_dtype(v[c]) and c not in skip:
                    v[c] = v[c] * k1 + k0
                elif c in ("text", "note", "name") and v[c].dtype == object:
                    v[c] = v[c].map(lambda x: shift_text(x) if isinstance(x, str) else x)
        elif isinstance(v, dict):
            v = copy.deepcopy(v)
        u[k] = v
    return u


def static_decimals(t: dict) -> set:
    """Десятичные числа, не изменившиеся при двух разных искажениях таблиц (т. е. не из таблиц), без нулевых."""
    orig = A.claims_fail
    A.claims_fail = lambda _t: []
    try:
        a, b, c = tokens(A.build(t)), tokens(A.build(perturb(t))), tokens(A.build(perturb(t, 0.77, 0.0913)))
    finally:
        A.claims_fail = orig
    return {x for x, y, z in zip(a, b, c) if x == y == z and float(x.replace(",", ".")) != 0.0} if len(a) == len(b) == len(c) else set(a)


def check_static(t: dict) -> list:
    extra = static_decimals(t) - TX.STATIC_DECIMALS
    return ["G50: числа вне таблиц и вне списка определений: %s" % sorted(extra)] if extra else []


def count_numbers(t: dict) -> int:
    return len(tokens(A.build(t)))
