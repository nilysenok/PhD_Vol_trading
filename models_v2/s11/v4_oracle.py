"""V4-A, G7: oracle bound (Ж-222 п. 2) — ТОЛЬКО диагностика на готовых прогнозах: использует реализованную RV цели, поэтому в прогноз не идёт.
Этот модуль не импортируется ни путями прогноза (`models_v2/s10`, остальные модули `s11`), ни мета-слоем (тест `test_v2_v4_isolation`).
QLIKE — `models_v2.metrics.qlike` (одно определение метрики, Д7).
O1 — лучший постоянный вес w ∈ [0, 1] задним числом (арифметический и геометрический); O2 — F = clip(RV, min(H, M), max(H, M)); O3 — O2 после одного множителя b = mean(RV/O2) задним числом
(b минимизирует QLIKE: d/db [RV/(b·x) + ln b] = 0); O4 — оболочка {H, K2, K3}: clip(RV, min3, max3). Части выигрыша лучшей одиночной модели: выбор веса, условный вес, выход за диапазон, остаток O3."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from models_v2 import metrics

GRID = np.linspace(0.0, 1.0, 101)


def _clip(x, lo, hi):
    """Крючок формулы oracle: тест с известным ответом подменяет его (мутация «oracle без clip»)."""
    return np.minimum(np.maximum(x, lo), hi)


def q(rv, f) -> float:
    return float(np.mean(metrics.qlike(np.asarray(rv, float), np.asarray(f, float))))


def mix(h, m, w: float, geom: bool):
    return np.exp((1.0 - w) * np.log(h) + w * np.log(m)) if geom else (1.0 - w) * h + w * m


def best_weight(rv, h, m, geom: bool) -> tuple:
    """-> (w, QLIKE) лучшего постоянного веса на [0, 1]: сетка 101 точка + уточнение ограниченной минимизацией вокруг лучшей точки; концы входят."""
    vals = np.array([q(rv, mix(h, m, w, geom)) for w in GRID])
    i = int(vals.argmin())
    lo, hi = GRID[max(i - 1, 0)], GRID[min(i + 1, len(GRID) - 1)]
    r = minimize_scalar(lambda w: q(rv, mix(h, m, w, geom)), bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})
    return (float(r.x), float(r.fun)) if r.fun < vals[i] else (float(GRID[i]), float(vals[i]))


def o2_series(rv, h, m):
    return _clip(np.asarray(rv, float), np.minimum(h, m), np.maximum(h, m))


def o3_series(rv, x):
    """O2 × b, b = mean(RV/x) — оптимальный множитель смещения задним числом (b = 1 допустим, поэтому QLIKE(O3) <= QLIKE(O2))."""
    b = float(np.mean(np.asarray(rv, float) / x))
    return b * x, b


def o4_series(rv, h, m2, m3):
    lo, hi = np.minimum(np.minimum(h, m2), m3), np.maximum(np.maximum(h, m2), m3)
    return _clip(np.asarray(rv, float), lo, hi)


def oracle(rv, h, m, third=None) -> dict:
    """Все O и разложение для пары (H, M); `third` — второй бустинг для O4 (оболочка трёх)."""
    rv, h, m = (np.asarray(x, float) for x in (rv, h, m))
    if len(rv) == 0 or not all(np.isfinite(x).all() and (x > 0).all() for x in (rv, h, m)):
        raise ValueError("oracle: пустая лента или RV/прогноз не конечны и не > 0")
    qh, qm = q(rv, h), q(rv, m)
    wa, o1a = best_weight(rv, h, m, False)
    wg, o1g = best_weight(rv, h, m, True)
    x2 = o2_series(rv, h, m)
    x3, b = o3_series(rv, x2)
    out = {"q_h": qh, "q_m": qm, "q_best": min(qh, qm), "o1_arith": o1a, "w_arith": wa, "o1_geom": o1g, "w_geom": wg, "o1": min(o1a, o1g, qh, qm), "o2": q(rv, x2), "o3": q(rv, x3), "b_mult": b}
    if third is not None:
        out["o4"] = q(rv, o4_series(rv, h, m, np.asarray(third, float)))
    return out | parts(out)


def parts(o: dict) -> dict:
    """Части выигрыша лучшей одиночной (сумма = q_best): p1 выбор постоянного веса, p2 условный вес, p3 выход за диапазон, p4 остаток O3 (нужна новая информация)."""
    p = {"p1_const_weight": o["q_best"] - o["o1"], "p2_conditional": o["o1"] - o["o2"], "p3_out_of_range": o["o2"] - o["o3"], "p4_residual": o["o3"]}
    s = o["q_best"]
    return p | {"s" + k[1:]: v / s for k, v in p.items()}


def known_answer_ok() -> bool:
    """Синтетика с известным ответом: H = 1, M = 3, RV = (2, 5, 0.5, 3). O2 = (2, 3, 1, 3): день внутри [1, 3] — потери 0; RV = 5 > 3 -> F = 3; RV = 0.5 < 1 -> F = 1."""
    rv, h, m = np.array([2.0, 5.0, 0.5, 3.0]), np.ones(4), 3.0 * np.ones(4)
    x2 = o2_series(rv, h, m)
    want = np.array([2.0, 3.0, 1.0, 3.0])
    ql = lambda r, f: r / f - np.log(r / f) - 1.0
    exp_q = float(np.mean([ql(5.0, 3.0), ql(0.5, 1.0)] + [0.0, 0.0]))
    return bool(np.allclose(x2, want, rtol=0, atol=1e-15) and abs(q(rv, x2) - exp_q) < 1e-12)
