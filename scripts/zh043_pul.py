"""Пул кандидатов плато (Ж-043, исправление дефекта пула Ж-042, PR #340): пул — ВСЕ посчитанные допустимые точки с эталоном s9,
включая достроенных соседей. Соседи новых лидеров достраиваются, пока топ не стабилизируется (не больше MAX_REP повторов);
все пробы остаются в N. Чистые функции: оценщик и соседи передаются снаружи (в тесте — синтетика).
"""
from __future__ import annotations

import math

MAX_REP, TOP_N, MIN_NB = 2, 10, 3


def fin(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def eligible(r: dict) -> bool:
    """Допустима: цель определена (mSR обоих лет) и активность ≥ 0,5 базы."""
    return fin(r["obj"]) and bool(r["aktivnost_ok"])


def top_of(rows: dict, n: int = TOP_N) -> list:
    """Топ-n по цели из ВСЕХ строк пула (факторные, OAT и соседи), без деления по роли. Точки с одинаковой подписью результата
    (`sig`: параметр не действует, результат тот же) — одна точка: в топ идёт первая."""
    out, seen = [], set()
    for r in sorted((r for r in rows.values() if eligible(r)), key=lambda r: -r["obj"]):
        if r.get("sig") is None or r["sig"] not in seen:
            out.append(r)
            seen.add(r.get("sig"))
    return out[:n]


def build_pool(rows: dict, evaluate, neighbors, key, n: int = TOP_N, max_rep: int = MAX_REP) -> list:
    """rows {key: строка с 'p','obj','aktivnost_ok','rol'} — посчитанные пробы; evaluate(p, rol) -> строка; neighbors(p) -> [p].

    Круг: топ-n по пулу; у членов топа без соседей достраиваются соседи (рол 'sosed'). Повтор, пока в топ не входит
    член без соседей; всего круг + max_rep повторов. Возвращает итоговый топ с плато (plato, sosedey, sosedey_pos, nb)."""
    for _ in range(max_rep + 1):
        new = [r for r in top_of(rows, n) if "nb" not in r]
        if not new:
            break
        for r in new:
            ks = []
            for q in neighbors(r["p"]):
                k = key(q)
                if k not in rows:
                    rows[k] = evaluate(q, "sosed")
                if rows[k].get("sig") is None or rows[k]["sig"] != r.get("sig"):    # сосед с тем же результатом — тот же дизайн, не сосед
                    ks.append(k)
            r["nb"] = ks
    top = top_of(rows, n)
    for r in top:
        nb = [rows[k] for k in r.get("nb", [])]
        v = [x["obj"] for x in nb if fin(x["obj"])]
        r["plato"] = sum([r["obj"], *v]) / (1 + len(v))
        r["sosedey"], r["sosedey_pos"] = len(v), sum(x > 0 for x in v)
        r["kray"] = int("nb" not in r or len(v) < MIN_NB)
    return top
