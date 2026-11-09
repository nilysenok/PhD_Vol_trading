"""V6-0: меры по ряду SBER. QLIKE — `models_v2.metrics.qlike`; RMSE/MAE/R² — `v4_b_stats.metr` (определение проекта, Д7); bias = mean ln(F/RV); окна по году цели t⁺.
`crit_SBER` = `d_comp.crit_wf` по годовым QLIKE целей 2017–2021 (то же определение, что `crit_panel` V5 для одной бумаги: бумага с < 2 годами не входит). EqY — равновесное среднее годовых QLIKE.
Условия «go» 2–4: без целей 2017; без 2017–2018; без 1 % дат окна выбора с наибольшей |QLIKE(ряд) − QLIKE(лучший из K2/K3 по дате)| (как V4-E, `v4_e_go.crit_trim`)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s6.xgb.d_comp import crit_wf
from models_v2.s11 import v4_b_stats as ST

YEARS = list(range(2017, 2022))
WIN = {**{str(y): (y, y) for y in range(2017, 2026)}, "2026": (2026, 2026), "2017_2021": (2017, 2021), "2022_2025": (2022, 2025), "all": (2017, 2026)}
MARGIN = 0.0005
TRIM = 0.01


def ql(rv, f) -> np.ndarray:
    return np.asarray(metrics.qlike(np.asarray(rv, float), np.asarray(f, float)), float)


def wmask(tp, w: str) -> np.ndarray:
    y = pd.DatetimeIndex(tp).year.to_numpy()
    lo, hi = WIN[w]
    return (y >= lo) & (y <= hi)


def metr_row(rv, f) -> dict:
    rv, f = np.asarray(rv, float), np.asarray(f, float)
    m = ST.metr(rv, f)
    return {**m, "bias_ln": float(np.mean(np.log(f / rv)))}


def annual(q, tp, years=YEARS) -> dict:
    y = pd.DatetimeIndex(tp).year.to_numpy()
    q = np.asarray(q, float)
    return {k: float(q[y == k].mean()) for k in years if (y == k).any()}


def crit_sber(q, tp, first_year: int = 2017) -> float:
    """crit_wf по целям first_year…2021; NaN при < 2 годов."""
    y = pd.DatetimeIndex(tp).year.to_numpy()
    m = (y >= first_year) & (y <= 2021)
    if len(set(y[m])) < 2:
        return float("nan")
    return float(crit_wf(np.asarray(q, float)[m], pd.Series(pd.DatetimeIndex(np.asarray(tp)[m]))))


def crit_trim(q, tp, qbest, frac: float = TRIM) -> float:
    """crit_SBER без доли `frac` дат окна 2017–2021 с наибольшей |q − qbest| (отбрасывание — своё для каждого ряда)."""
    y = pd.DatetimeIndex(tp).year.to_numpy()
    m = (y >= 2017) & (y <= 2021)
    q, tpw, qb = np.asarray(q, float)[m], np.asarray(tp)[m], np.asarray(qbest, float)[m]
    drop = np.argsort(-np.abs(q - qb), kind="stable")[: int(round(frac * len(q)))]
    keep = np.ones(len(q), bool)
    keep[drop] = False
    return float(crit_wf(q[keep], pd.Series(pd.DatetimeIndex(tpw[keep]))))


def years_better(q, tp, q_k2, q_k3) -> int:
    """Число лет из 2017–2021, где средний QLIKE ряда < лучшего (меньшего) из годовых QLIKE K2 и K3 того же года."""
    a, b, c = annual(q, tp), annual(q_k2, tp), annual(q_k3, tp)
    return int(sum(a[y] < min(b[y], c[y]) for y in YEARS if y in a))


def crit_row(rv, f, tp, q_k2, q_k3) -> dict:
    """Все меры отбора ряда на датах tp: crit_SBER, без 2017, без 2017–2018, без 1 % крайних, EqY, лет лучше, годовые QLIKE."""
    q = ql(rv, f)
    qbest = np.minimum(q_k2, q_k3)
    ann = annual(q, tp)
    return {"crit": crit_sber(q, tp, 2017), "crit_no2017": crit_sber(q, tp, 2018), "crit_no2017_18": crit_sber(q, tp, 2019), "crit_trim1": crit_trim(q, tp, qbest),
            "EqY": float(np.mean(list(ann.values()))), "years_better_of5": years_better(q, tp, q_k2, q_k3), **{"q%d" % y: v for y, v in ann.items()}}
