"""V5-0, G2/G3: разбор весов годов в `crit_panel` (Ж-217 = среднее по M16 `d_comp.crit_wf`) и равновесная по годам оценка EqY (диагностика, не выбор).
Механизм (d_comp.py:14-24): ann_y — средний QLIKE бумаги по году цели; для Y = 2018…2021 pen_Y = mean_{y<Y} ann_y + 0,25·(max − mean); crit_wf = mean_Y pen_Y. Год Y в pen_Y не входит, поэтому 2021 весит 0, а 2017 входит во все четыре pen_Y.
Здесь — аналитическая формула по годовым QLIKE бумаг (расхождение с `crit_panel` ≤ 1e-12), веса, покрытие дат, EqY. `crit_panel` и `crit_wf` V4 не меняются; ошибки реализации не найдено, версионированной замены нет."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s6.xgb.d_comp import PENALTY
from models_v2.s10 import hp_stats_sel as S
from models_v2.s12 import v5_0_io as IO

YEARS = list(range(S.CRIT_YEARS[0], S.CRIT_YEARS[1] + 1))
TOL = 1e-12


def ql(rv, f) -> np.ndarray:
    r = np.asarray(rv, float) / np.asarray(f, float)
    return r - np.log(r) - 1.0


def annual(k: pd.DataFrame, cols: list) -> pd.DataFrame:
    """-> (series, ticker, year, ann, n): средний QLIKE бумаги M16 по году цели на ОБЩИХ датах окна выбора (как `S.crit_panel`); бумага с < 2 годами не входит."""
    w = S.crit_window(S.common(k, cols))
    y = w["t_plus"].dt.year.to_numpy()
    rows = []
    for c in cols:
        g = pd.DataFrame({"ticker": w["ticker"].to_numpy(), "year": y, "q": ql(w["rv_tplus"], w[c])}).groupby(["ticker", "year"])["q"].agg(["mean", "size"]).reset_index()
        g.insert(0, "series", c)
        rows.append(g.rename(columns={"mean": "ann", "size": "n"}))
    a = pd.concat(rows, ignore_index=True)
    ny = a.groupby(["series", "ticker"])["year"].transform("nunique")
    return a[ny >= 2].reset_index(drop=True)


def pen_parts(ann: np.ndarray) -> tuple:
    """Годовые QLIKE бумаги по порядку лет -> (база Σ w_y·ann_y, штраф, crit) по формуле: база = mean_Y mean(past), штраф = 0,25·mean_Y (max − mean)."""
    ann = np.asarray(ann, float)
    base, pen = [], []
    for i in range(1, len(ann)):
        m = ann[:i].mean()
        base.append(m)
        pen.append(PENALTY * (ann[:i].max() - m))
    return float(np.mean(base)), float(np.mean(pen)), float(np.mean(base) + np.mean(pen))


def weights(n_years: int) -> np.ndarray:
    """Эффективные веса лет в базовой части: w_y = (1/(n−1))·Σ_{Y>y} 1/|{y' < Y}|; 2-й с конца… последний год весит 0."""
    w = np.zeros(n_years)
    for Y in range(1, n_years):
        w[:Y] += 1.0 / Y
    return w / (n_years - 1)


def crit_analytic(a: pd.DataFrame) -> pd.DataFrame:
    """-> (series, crit_analytic, base_part, penalty_part, n_tickers): среднее по бумагам аналитической формулы."""
    rows = []
    for s, g in a.groupby("series", sort=False):
        per = [pen_parts(t.sort_values("year")["ann"].to_numpy()) for _, t in g.groupby("ticker")]
        b, p, c = (float(np.mean([x[i] for x in per])) for i in range(3))
        rows.append(dict(series=s, crit_analytic=c, base_part=b, penalty_part=p, n_tickers=len(per)))
    return pd.DataFrame(rows)


def audit_tables(k: pd.DataFrame, cols: list) -> tuple:
    """-> (CRIT_AUDIT, вердикт-словарь, annual): (а) воспроизведение crit_panel; (б) веса; (в) покрытие; (г/д) вывод."""
    a = annual(k, cols)
    cp = S.crit_panel(k, cols).set_index("series")["crit_panel"]
    an = crit_analytic(a).set_index("series")
    ser = an.assign(crit_panel=cp.reindex(an.index)).reset_index()
    ser["abs_diff"] = (ser["crit_panel"] - ser["crit_analytic"]).abs()
    ser.insert(0, "section", "series")
    ser = ser.rename(columns={"series": "item"})
    w = weights(len(YEARS))
    wt = pd.DataFrame({"section": "weight", "item": "w_base_part", "year": YEARS, "weight": w})
    raw = k[(k["group"] == "M16") & k["t_plus"].dt.year.between(YEARS[0], YEARS[-1]) & np.isfinite(k["rv_tplus"])]
    cov = raw.assign(year=raw["t_plus"].dt.year).groupby(["ticker", "year"]).size().rename("n_raw").reset_index()
    cm = a[a["series"] == "k1"][["ticker", "year", "n"]].rename(columns={"n": "n_target"})
    cov = cov.merge(cm, on=["ticker", "year"], how="left").fillna({"n_target": 0})
    cov["note"] = "пропуск = n_raw − n_target"
    cov.insert(0, "section", "coverage")
    cov = cov.rename(columns={"ticker": "item"})
    own = S.crit_panel(k, ["k1"]).iloc[0]["crit_panel"]
    ver = dict(max_abs_diff=float(ser["abs_diff"].max()), w=w, years_all=bool((a.groupby(["series", "ticker"])["year"].nunique() == len(YEARS)).all()),
               n_tickers=int(a["ticker"].nunique()), missing_share=float(1 - cov["n_target"].sum() / cov["n_raw"].sum()),
               k1_common=float(cp["k1"]), k1_own_dates=float(own), w2017=float(w[0]), w2021=float(w[-1]), kind="определение")
    return pd.concat([ser, wt, cov], ignore_index=True), ver, a


def eqy_table(a: pd.DataFrame, cp: pd.Series, stage: dict) -> tuple:
    """CRIT_EQY: QLIKE M16 по годам 2017…2021, EqY = mean по годам, crit_panel, ранги (по 46 + 5 контролей), Δ к лучшему из K2/K3 в % уровня K1 (по EqY и по crit_panel)."""
    p = a.groupby(["series", "year"])["ann"].mean().unstack("year")
    p = p[YEARS]
    t = p.copy()
    t.columns = ["qlike_%d" % y for y in YEARS]
    t["EqY"] = p.mean(axis=1)
    t["crit_panel"] = cp.reindex(t.index)
    t["rank_EqY"] = t["EqY"].rank(method="min").astype(int)
    t["rank_crit"] = t["crit_panel"].rank(method="min").astype(int)
    best_e, best_c = min(t.loc["k2", "EqY"], t.loc["k3", "EqY"]), min(t.loc["k2", "crit_panel"], t.loc["k3", "crit_panel"])
    t["d_EqY_best_k23_pct_k1"] = 100.0 * (t["EqY"] - best_e) / t.loc["k1", "EqY"]
    t["d_crit_best_k23_pct_k1"] = 100.0 * (t["crit_panel"] - best_c) / t.loc["k1", "crit_panel"]
    t.insert(0, "stage", [stage.get(s, "контроль") for s in t.index])
    return t.reset_index().rename(columns={"index": "series"}), p
