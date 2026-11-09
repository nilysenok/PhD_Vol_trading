"""V6 лист Z3, G68: сверка документа причин (ТОЛЬКО ЧТЕНИЕ). Документ на диске = пересборка из таблиц на диске, число за числом (десятичные числа попарно, весь текст побайтно).
Статические части не содержат результатных чисел: `static_decimals` искажает таблицы и оставляет числа, не зависящие от них; список разрешённых пуст — любое такое число в тексте есть нарушение."""
from __future__ import annotations

import re

import pandas as pd

from models_v2.s13 import v6_why_data as DT, v6_why_doc as W0, v6_why_doc2 as W

DEC = re.compile(r"\d+,\d+")
PARTS = ["1. Годы", "2. Крайние дни", "3. Режимы", "4. Калибровка", "5. Сколько бустинговой", "6. Устойчивость", "7. Величина поправки", "Что не проверено"]
SKIP = {"frac"}


def tokens(txt: str) -> list:
    return DEC.findall(txt)


def compare(doc: str, fresh: str) -> tuple:
    """-> (сверено чисел, расхождений): попарно по порядку; разная длина — лишние числа считаются расхождениями."""
    a, b = tokens(doc), tokens(fresh)
    return len(b), sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))


def check_doc(t: dict | None = None, doc: str | None = None) -> list:
    t = W0.load_tables() if t is None else t
    if doc is None and not DT.DOC.exists():
        return ["G68: нет файла %s" % DT.DOC.name]
    doc = DT.DOC.read_text(encoding="utf-8") if doc is None else doc
    try:
        fresh = W.build(t)
    except (KeyError, ValueError) as e:
        return ["G68: таблицы на диске не собирают документ: %s" % e]
    n, bad = compare(doc, fresh)
    out = []
    if bad or doc != fresh:
        out.append("G68: документ ≠ пересборке из таблиц (чисел %d, расхождений %d)" % (n, bad))
    if doc.splitlines()[0] != W.first_line(t) or not doc.startswith("Вывод") or "диагностика, не отбор" not in doc[:2500]:
        out.append("G68: первая строка / шапка")
    heads = [ln for ln in doc.splitlines() if ln.startswith("## ")]
    out += ["G68: нет раздела %d «%s»" % (i + 1, p) for i, p in enumerate(PARTS) if len(heads) <= i or p not in heads[i]]
    if "bps" in doc.lower() or "V7" in doc:
        out.append("G68: bps или V7 в документе (владельцу — проценты; рекомендаций по V7 нет)")
    return out


def perturb(t: dict, k1: float = 1.31, k0: float = 0.1371) -> dict:
    """Копия таблиц: каждое вещественное число ×1,31 + 0,1371 (кроме `frac`)."""
    u = {}
    for k, v in t.items():
        v = v.copy()
        for c in v.columns:
            if pd.api.types.is_float_dtype(v[c]) and c not in SKIP:
                v[c] = v[c] * k1 + k0
        u[k] = v
    return u


def static_decimals(t: dict) -> set:
    """Десятичные числа, не изменившиеся при двух разных искажениях таблиц (т. е. не из таблиц), без нулевых."""
    a, b, c = tokens(W.build(t)), tokens(W.build(perturb(t))), tokens(W.build(perturb(t, 0.77, 0.0913)))
    return {x for x, y, z in zip(a, b, c) if x == y == z and float(x.replace(",", ".")) != 0.0} if len(a) == len(b) == len(c) else set(a)


def check_static(t: dict) -> list:
    try:
        extra = static_decimals(t)
    except (KeyError, ValueError):
        return []                                                      # причина названа в check_doc
    return ["G68: числа вне таблиц: %s" % sorted(extra)] if extra else []


def count_numbers(t: dict) -> int:
    return len(tokens(W.build(t)))
