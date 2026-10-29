"""Таблица §6 предрегистрации v2 (черновик): Δ, p Ледуа–Вольфа, Холм. Формат `v: "1"`.

Δ = SR(a) − SR(b): разность табличных дневных Шарпов (ddof=1, ·√252) на общих OOS-днях
(мутация № 13: не из бутстрапа ЛВ); ЛВ даёт только p (блок 5, 10 000 повторов, seed 42, двусторонний).
Список сравнений и m — параметры (❓4): m = len(pary), по умолчанию PARY_V2 (m = 5, Ж-114 п. 7).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from fastbt.s9 import _holm
from fastbt.s9_stats import lw_sharpe_diff

V = "1"
ALPHA, BLOCK, REPS, SEED = 0.05, 5, 10_000, 42
PARY_V2 = (("A'-базовая", "A_strogaya"), ("A'-расширенная", "A_strogaya"),
           ("C", "A'-базовая"), ("D", "A'-базовая"), ("МСП λ=1", "МСП λ=0"))


def sr_table(x: np.ndarray) -> float:
    """Табличный дневной Шарп: mean/sd (ddof=1)·√252; нулевая дисперсия — ошибка, не 0."""
    s = np.std(x, ddof=1)
    if not s > 1e-12:
        raise ValueError("нулевая дисперсия ряда: Шарп не определён")
    return float(np.mean(x) / s * math.sqrt(252))


def _obshchie(a: pd.Series, b: pd.Series, block: int) -> tuple[np.ndarray, np.ndarray]:
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if len(j) < 3 * block:
        raise ValueError(f"общих дней {len(j)} < {3 * block}: сравнение невозможно")
    return j["a"].to_numpy(float), j["b"].to_numpy(float)


def sravnit(ryady: dict[str, pd.Series], pary=PARY_V2, alpha=ALPHA, reps=REPS,
            block=BLOCK, seed=SEED) -> dict:
    """ryady: имя → дневная избыточная (индекс — даты). Возвращает таблицу v "1"."""
    if not pary:
        raise ValueError("пустой список сравнений")
    rows = []
    for na, nb in pary:
        for n in (na, nb):
            if n not in ryady:
                raise KeyError(f"нет ряда {n!r}")
        x, y = _obshchie(ryady[na], ryady[nb], block)
        sa, sb = sr_table(x), sr_table(y)
        rows.append(dict(a=na, b=nb, n_dney=len(x), sr_a=sa, sr_b=sb, delta=sa - sb,
                         p_lw=float(lw_sharpe_diff(x, y, reps, block, seed)[2])))
    for r, p in zip(rows, _holm(np.array([r["p_lw"] for r in rows]))):
        r["p_holm"], r["otkl"] = float(p), bool(p < alpha)
    return dict(v=V, m=len(rows), alpha=alpha, block=block, reps=reps, seed=seed, rows=rows)
