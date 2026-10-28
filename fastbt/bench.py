"""Бенчмарки для H3 (гл. 4, табл. 4.9): MCFTR с экспозицией МСП-A и 16%, остальное под
ключевую ставку (ежедневная ребалансировка); классические MCFTR/RGBITR 20/80, 60/40, 80/20
с ежемесячной ребалансировкой. Доходности логарифмические, ставка за календарные дни.

H3: Шарп(МСП-A) > Шарп(бенчмарка); парный бутстрап по общим дням (report.boot_delta).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fastbt import msp
from fastbt.data import MAIN
from fastbt.report import boot_delta

MCFTR = MAIN / "data" / "mcftr_iss.csv"
RGBITR = MAIN / "data" / "rgbitr_iss.csv"
CLASSIC = {"20/80": 0.2, "60/40": 0.6, "80/20": 0.8}


def _close(path) -> pd.Series:
    d = pd.read_csv(path, parse_dates=["date"])
    return d.set_index("date")["close"].sort_index()


def _frame(rate: pd.Series) -> pd.DataFrame:
    m, b = _close(MCFTR), _close(RGBITR)
    df = pd.DataFrame({"m": m, "b": b}).dropna()
    # доходность и дни ставки вперёд, d→d_next, как у МСП (r[d] = d→d+1, кэш за [d, d_next)):
    # экспозиция, выбранная на закрытии d, умножается на доходность d→d+1 (аудит FA, H3, 26.10)
    df["rm"] = np.log(df["m"]).diff().shift(-1)
    df["rb"] = np.log(df["b"]).diff().shift(-1)
    d = pd.Series(df.index, index=df.index)
    df["cd"] = (d.shift(-1) - d).dt.days
    kr = rate.reindex(rate.index.union(df.index)).sort_index().ffill().bfill().reindex(df.index)
    df["rf_cd"] = kr.values / 365.0 * df["cd"]
    return df.dropna()


def _classic(df: pd.DataFrame, w: float) -> np.ndarray:
    """Стоимость портфеля с ребалансировкой к w в первый торговый день месяца."""
    sm, sb = np.expm1(df["rm"].values), np.expm1(df["rb"].values)
    month = df.index.to_period("M")
    out = np.empty(len(df))
    wm, wb = w, 1.0 - w
    for i in range(len(df)):
        if i == 0 or month[i] != month[i - 1]:
            tot = wm + wb
            wm, wb = w * tot, (1.0 - w) * tot
        v0 = wm + wb
        wm, wb = wm * (1 + sm[i]), wb * (1 + sb[i])
        out[i] = np.log((wm + wb) / v0)
    return out


def bench_series(rate: pd.Series, exposure: pd.Series) -> dict:
    """{имя: DataFrame(total, excess)} на календаре индексов."""
    df = _frame(rate)
    e = exposure.reindex(exposure.index.union(df.index)).sort_index().ffill().reindex(df.index)
    out = {}
    w = e.fillna(0.0).values
    out["MCFTR, экспозиция МСП-A"] = w * df["rm"].values + (1 - w) * df["rf_cd"].values
    out["MCFTR 16% + ставка"] = 0.16 * df["rm"].values + 0.84 * df["rf_cd"].values
    for name, wc in CLASSIC.items():
        out[f"MCFTR/RGBITR {name}"] = _classic(df, wc)
    return {k: pd.DataFrame({"total": v, "excess": v - df["rf_cd"].values}, index=df.index)
            for k, v in out.items()}


def _metrics(fr: pd.DataFrame) -> dict:
    m = msp.metrics(fr.assign(exposure=fr.get("exposure", np.nan)))
    m.pop("exposure_pct", None)
    return m


def h3_table(port: pd.DataFrame, rate: pd.Series, reps: int, start, end,
             lw_block: int = 0) -> pd.DataFrame:
    """МСП-A (port из msp.portfolio) против бенчмарков на общих днях окна [start, end].
    lw_block > 0 (тест главы 4, владелец 26.10, 10:05, п. 12): p_boot — Ледуа–Вольф с этим блоком,
    IID — в p_boot_iid; по умолчанию — IID, как прежде."""
    p = port.loc[start:end]
    rows = [dict(portfolio="МСП-A (fastbt)", **msp.metrics(p))]
    for name, fr in bench_series(rate, port["exposure"]).items():
        fr = fr.loc[start:end]
        j = p.index.intersection(fr.index)
        xa, xb = p.loc[j, "excess"].values, fr.loc[j, "excess"].values
        obs, pv, lo, hi = boot_delta(xa, xb, reps)
        m = _metrics(fr.loc[j])
        m.update(portfolio=name, d_sharpe_msp_minus=obs, p_boot=pv, ci_lo=lo, ci_hi=hi,
                 common_days=len(j))
        if lw_block:
            from fastbt.s9_stats import lw_sharpe_diff
            m.update(p_boot=lw_sharpe_diff(xa, xb, reps, lw_block)[2], p_boot_iid=pv)
        rows.append(m)
    return pd.DataFrame(rows)
