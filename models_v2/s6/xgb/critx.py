"""Внутренний критерий волны B (GATES_B_xgb P1): годовой LOYO по прошлым целям + штраф за худший прошлый год.
Для года Y: pen_Y = mean_y q + 0,25·(max_y q − mean_y q), y ∈ [2015, Y−1]; crit_pen = mean_Y pen_Y; crit_mean = mean_Y mean_y q (= q_loyo Ж-204)."""
from __future__ import annotations

import pandas as pd

PENALTY = 0.25
YEARS = tuple(range(2017, 2027))


def crit(folds: pd.DataFrame, penalty: float = PENALTY) -> dict:
    """folds: колонки Y, y, q. Фолды с y >= Y (не прошлое) и NaN отбрасываются; год Y без фолдов — ошибка."""
    f = folds[folds["y"] < folds["Y"]].dropna(subset=["q"])
    g = f.groupby("Y")["q"]
    mean, worst = g.mean(), g.max()
    if tuple(mean.index) != YEARS:
        raise ValueError("годы критерия: %s" % list(mean.index))
    pen = mean + penalty * (worst - mean)
    return {"crit_mean": float(mean.mean()), "crit_pen": float(pen.mean()), "by_year": {int(k): float(v) for k, v in pen.items()}}


def better(a: float, b: float, tol: float = 1e-12) -> bool:
    """Критерий a строго лучше b (меньше): на > tol."""
    return bool(b - a > tol)


def perspective(crit_pen: float, crit_pen_base: float, d_harj_no_240222: float, unstable: bool) -> bool:
    """P2: crit_pen лучше baseline XGBoost И разность к HAR-J(б) без 24.02.2022 <= 0 И прогноз устойчив."""
    return bool(better(crit_pen, crit_pen_base) and d_harj_no_240222 <= 0 and not unstable)
