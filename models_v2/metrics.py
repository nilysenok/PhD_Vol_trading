"""Метрики и срезы (Д7: одно определение на проект, Ж-197 О7–О8).

QLIKE = r - ln r - 1, r = RV/sigma2; mean_r = среднее r (смещение); RMSE/MAE — ошибки e = ln RV - ln sigma2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_RVI_PRIOR = 20


def qlike(rv, s2) -> np.ndarray:
    r = np.asarray(rv, float) / np.asarray(s2, float)
    return r - np.log(r) - 1.0


def summary(rv, s2) -> dict:
    rv, s2 = np.asarray(rv, float), np.asarray(s2, float)
    if len(rv) == 0:
        raise ValueError("пустой срез")
    e = np.log(rv) - np.log(s2)
    return {"n": len(rv), "qlike": float(qlike(rv, s2).mean()), "mean_r": float((rv / s2).mean()),
            "rmse_ln": float(np.sqrt((e ** 2).mean())), "mae_ln": float(np.abs(e).mean())}


def rvi_regime(d_dates: pd.Series, rvi: pd.DataFrame) -> pd.Series:
    """Режим по дню прогноза d: 'high' если RVI_d > медианы RVI по датам < d (расширяющейся), иначе 'low'.
    Первые 20 дат RVI и даты без RVI — NaN."""
    s = rvi.set_index("date")["close_RVI"].sort_index()
    med = pd.Series([s.iloc[:k].median() if k >= MIN_RVI_PRIOR else np.nan for k in range(len(s))], index=s.index)
    reg = pd.Series(np.where(s > med, "high", "low"), index=s.index).where(med.notna())
    return d_dates.map(reg)


def slice_masks(f: pd.DataFrame) -> dict:
    """f: строки одной модели с колонками d, target_date, regime. Срезы по ДАТЕ ЦЕЛИ."""
    t = f["target_date"]
    m = {"all": np.ones(len(f), bool)}
    for y in range(2017, 2027):
        m["y%d" % y] = (t.dt.year == y).to_numpy()
    ok = (t.dt.year <= 2021).to_numpy()
    m["rvi_high"] = ((f["regime"] == "high").to_numpy()) & ok
    m["rvi_low"] = ((f["regime"] == "low").to_numpy()) & ok
    m["mar_jun_2020"] = ((t >= "2020-03-01") & (t <= "2020-06-30")).to_numpy()
    m["y2022"] = (t.dt.year == 2022).to_numpy()
    return m


def metrics_table(fc: pd.DataFrame, rvi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, f in fc.groupby("model", sort=False):
        f = f.sort_values("d").reset_index(drop=True)
        f["regime"] = rvi_regime(f["d"], rvi)
        for name, mk in slice_masks(f).items():
            if mk.sum() == 0:
                continue
            note = "неполный (янв–фев)" if name == "y2026" else ""
            rows.append({"model": model, "slice": name, **summary(f["rv"][mk], f["sigma2"][mk]), "note": note})
    return pd.DataFrame(rows)
