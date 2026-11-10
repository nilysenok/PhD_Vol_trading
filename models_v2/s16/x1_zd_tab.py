"""ZD: таблицы по готовым D и помесячным прогнозам (`x1_zd_core.walk`). Периоды: годы t⁺ 2018…2026 и окна `all`, `2018-2021`, `2022-2025`. Файлов не пишет."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

YEARS = list(range(2018, 2027))
WINDOWS = {"all": (2018, 2026), "2018-2021": (2018, 2021), "2022-2025": (2022, 2025)}
PERIODS = [str(y) for y in YEARS] + list(WINDOWS)


def mask(year: np.ndarray, period: str) -> np.ndarray:
    if period in WINDOWS:
        lo, hi = WINDOWS[period]
        return (year >= lo) & (year <= hi)
    return year == int(period)


def auc(score, pos) -> float:
    """AUC (Манн–Уитни, ничьи — средние ранги): вероятность, что у «положительного» балл выше. Один класс → NaN."""
    score, pos = np.asarray(score, float), np.asarray(pos, bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def states_table(Ds: dict, e: np.ndarray, year: np.ndarray) -> pd.DataFrame:
    """D по состоянию E (e_d = 1) / N (0) / all × период: n, сумма, среднее. Строки 2018-го года и всех дат (Y1, Y2, Y3 на общих n = 2018)."""
    rows = []
    for y, D in Ds.items():
        for p in PERIODS:
            for st, sm in (("E", e == 1), ("N", e == 0), ("all", np.ones(len(e), bool))):
                m = mask(year, p) & sm
                rows.append((y, st, p, int(m.sum()), float(D[m].sum()), float(D[m].mean()) if m.any() else float("nan")))
    return pd.DataFrame(rows, columns=["y", "state", "period", "n", "sum_D", "mean_D"])


def identity(t: pd.DataFrame, Ds: dict) -> list:
    """Σ D по состояниям = Σ по годам = Σ всего (и = Σ D исходного ряда). -> список нарушений (пусто = 0)."""
    bad = []
    for y, D in Ds.items():
        g = t[t["y"] == y]
        by_year = g[(g["state"] == "all") & g["period"].isin([str(v) for v in YEARS])]["sum_D"].sum()
        by_state = g[(g["state"].isin(["E", "N"])) & g["period"].isin([str(v) for v in YEARS])]["sum_D"].sum()
        tot = g[(g["state"] == "all") & (g["period"] == "all")]["sum_D"].iloc[0]
        n_ok = g[(g["state"] == "all") & g["period"].isin([str(v) for v in YEARS])]["n"].sum() == len(D)
        if not (n_ok and np.isclose(by_year, tot) and np.isclose(by_state, tot) and np.isclose(tot, D.sum())):
            bad.append("%s: Σ по состояниям %.9g, по годам %.9g, всего %.9g, ряд %.9g" % (y, by_state, by_year, tot, D.sum()))
    return bad


def pred_table(preds: dict, Ds: dict, e: np.ndarray, year: np.ndarray, act: np.ndarray) -> pd.DataFrame:
    """Метрики по периодам (только даты после включения): AUC (а), (в), (г) против 1[D<0]; ρ (б) с D; Y2 — средний D в верхней/нижней половине прогноза (б)."""
    rows = []
    for y, D in Ds.items():
        pr = preds[y]
        for p in PERIODS:
            m = act & mask(year, p)
            Dm, pos = D[m], D[m] < 0
            row = dict(y=y, period=p, n=int(m.sum()), share_neg=float(pos.mean()) if m.any() else float("nan"))
            row.update(auc_a=auc(pr["pa"][m], pos), auc_c=auc(pr["pc"][m], pos), auc_g=auc(pr["pg"][m], pos), rho_b=spearman(pr["pb"][m], Dm))
            hi = m & (pr["pb"] >= np.median(pr["pb"][m])) if m.any() else m
            lo = m & ~hi
            row.update(D_hi=float(D[hi].mean()) if y == "Y2" and hi.any() else float("nan"), D_lo=float(D[lo].mean()) if y == "Y2" and lo.any() else float("nan"))
            rows.append(row)
    return pd.DataFrame(rows)


def calib_table(pa: np.ndarray, D: np.ndarray, act: np.ndarray) -> pd.DataFrame:
    """Децили прогноза (а) на весь период после включения (Y2): средний прогноз и доля D<0."""
    idx = np.flatnonzero(act)
    idx = idx[np.argsort(pa[idx], kind="stable")]
    return pd.DataFrame([(k + 1, len(c), float(pa[c].mean()), float((D[c] < 0).mean())) for k, c in enumerate(np.array_split(idx, 10))], columns=["decile", "n", "mean_pred", "share_neg"])
