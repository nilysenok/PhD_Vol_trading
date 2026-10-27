"""Ансамбль по плато (Ж-051, итерация 2): чистая часть без данных — набор точек, сведение позиций, плато и матрица.

Определение (умолчание дирижёра, класс B, блок 01:05 не дошёл):
 - центр — последняя строка S{k}_itog_v2.csv; соседи — только по осям набора, радиус 1 (до 2k+1 точек, у края оси
   сосед один); углы куба не считаются; точки с событиями div/cbr_div (не считаются, ❓5) пропускаются;
 - точки, различающиеся только осями исполнения (k_s, R, τ, σ, вход, размер…), дают тот же сигнал: ансамбль
   схлопывает их по ключу `signal_key` — выходы и размер у ансамбля всегда центральной точки;
 - режим `srednee`: m = среднее позиций точек; направление sign(m), вес сделки умножается на |m| (дневной
   вес фиксируется на сделку движком этапа 1); режим `golos`: строгое большинство точек (> n/2) в одну сторону,
   иначе 0, вес 1. Ансамбль из одной точки равен этой точке побайтно (golden-тест).
Эта часть не читает данных; сборка позиций точек (боевой путь `ekviv_s9`/`Evaluator._positions`) и прогон — в драйвере.
"""
from __future__ import annotations

import math

import numpy as np

MODES = ("srednee", "golos")
NABORY = ("top2", "top3", "vse")
NE_SCHITANO = ("div", "cbr_div")
SIGNAL_EXTRA = ("q_sily", "rezhim", "sobytiya")      # оси вне sig_names, которые меняют, какие сделки Sk берутся
PLATO_NET = 0.1                                       # как в Ж-048: сосед не хуже лучшей более чем на 0,1 mSR


def osi_nabora(nabor: str, ranked: list, axes: dict) -> list:
    """top2/top3 — первые оси по μ* (Моррис, S{k}_morris.csv), vse — все оси сетки."""
    if nabor not in NABORY:
        raise ValueError(f"набор {nabor!r}: допустимо {NABORY}")
    return list(axes) if nabor == "vse" else [a for a in ranked if a in axes][:int(nabor[3:])]


def tochki(center: dict, osi: list, axes: dict) -> list:
    """Центр + соседи ±1 по каждой оси из osi (по одной оси за раз). Без дублей; события div/cbr_div пропущены."""
    out, seen = [dict(center)], {tuple(sorted(center.items()))}
    for a in osi:
        j = axes[a].index(center[a])
        for jj in (j - 1, j + 1):
            if not 0 <= jj < len(axes[a]):
                continue
            q = {**center, a: axes[a][jj]}
            k = tuple(sorted(q.items()))
            if k in seen or q.get("sobytiya") in NE_SCHITANO:
                continue
            seen.add(k)
            out.append(q)
    return out


def signal_key(pt: dict, sig_keys) -> tuple:
    """Всё, что меняет позиции точки до движка; оси исполнения (выходы, σ, вход, размер) в ключ не входят."""
    return tuple((k, pt[k]) for k in (*sorted(sig_keys), *SIGNAL_EXTRA) if k in pt)


def signal_points(pts: list, sig_keys) -> list:
    """Различные по сигналу точки в порядке первого появления (центр — первый)."""
    seen, out = set(), []
    for p in pts:
        k = signal_key(p, sig_keys)
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def combine(P: np.ndarray, mode: str) -> tuple:
    """P[n_точек, n_дней] ∈ [−1; 1] → (направление ∈ {−1,0,1}, вес-множитель ∈ [0;1]). День d — только столбец d."""
    P = np.asarray(P, float)
    if P.ndim != 2 or not len(P):
        raise ValueError("P: нужна матрица [точки, дни] минимум с одной точкой")
    if mode == "srednee":
        m = P.mean(axis=0)
        return np.sign(m), np.abs(m)
    if mode == "golos":
        n = P.shape[0]
        d = np.where((P > 0).sum(0) > n / 2, 1.0, np.where((P < 0).sum(0) > n / 2, -1.0, 0.0))
        return d, (d != 0).astype(float)
    raise ValueError(f"режим {mode!r}: допустимо {MODES}")


def plato_ansambl(obj_cfg: dict, best: tuple) -> dict:
    """Плато лучшей конфигурации (набор, режим): соседи — тот же набор в другом режиме и соседние наборы в том же.
    Правило «≥ 3 соседей» невыполнимо (соседей 2–3): берём одностороннее правило Ж-048 — все соседи конечны, > 0 и
    не хуже лучшей более чем на PLATO_NET. Число соседей пишется в отчёт."""
    nab, mode = best
    i = NABORY.index(nab)
    nb = [(n, mode) for n in (NABORY[i - 1] if i else None, NABORY[i + 1] if i + 1 < len(NABORY) else None) if n]
    nb.append((nab, MODES[1 - MODES.index(mode)]))
    o = obj_cfg[best]
    ok = all(math.isfinite(obj_cfg[c]) and obj_cfg[c] > 0 and obj_cfg[c] >= o - PLATO_NET for c in nb)
    return dict(ok=bool(ok), n_sosedey=len(nb), sosedi=nb)


def cfg_key(nabor: str, mode: str) -> tuple:
    return ("ansambl", nabor, mode)


def dobavit_v_ledger(led, nabor: str, mode: str, res: dict) -> None:
    """Конфигурация ансамбля — столбец общей матрицы и +1 в N (Ж-051 п. 2). Кэш Ledger только дополняется."""
    k = (cfg_key(nabor, mode), "train")
    if k in led.cache:
        raise ValueError(f"конфигурация {k[0]} уже в матрице: реестр только дополняется")
    led.cache[k] = ({"ansambl": nabor, "rezhim_ansambl": mode}, res)
