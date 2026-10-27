"""Конвейер B/C/D: N проб (T3). Отдельный файл — лимит 200 строк у konveyer_bcd.py.

Граница окна: доходность последнего бара = закрытие по close первого бара после окна — правило Н5, общее с `s9`, T1 и
эталонами final3; конвейер его не меняет (решение за владельцем, расследование #317).

- N проб = всё посчитанное: и точки, выпавшие по экрану D (у них считается только Шарп и моменты), по каждой комиссии.
  DSR — `sweep_drivers.dsr` (дневной SR, kurt обычный), Холм — по p_vs_A всех проб подхода (у невычисленных p = 1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fastbt.s9 import _holm
from fastbt.sweep_drivers import dsr


def sr_stats(ex) -> dict:
    s = pd.Series(np.asarray(ex, float))
    sd = s.std(ddof=1)
    return dict(sr_d=float(s.mean() / sd) if sd > 0 else float("nan"), T=len(s), skew=float(s.skew()),
                kurt=float(s.kurt()) + 3.0)


def light_rows(eng, bases: dict, G, meta: dict) -> list:
    """Точка, выпавшая по экрану: строки по комиссиям с моментами (для N), без к2/ЛВ."""
    g = eng.g_row(G)
    return [dict(meta, comm_pct=round(c * 100, 6), **sr_stats(eng.port(b.A * g, comm=c)["excess"].values))
            for c, b in bases.items()]


def add_n(df: pd.DataFrame, declared: int | None = None) -> pd.DataFrame:
    """Колонки n_prob, dsr, holm_p_vs_A по каждой комиссии; declared — объявленное N (проверка по числу строк)."""
    df = df.copy()
    df["n_prob"], df["dsr"], df["holm_p_vs_A"] = np.nan, np.nan, np.nan
    for _, ix in df.groupby("comm_pct").groups.items():
        sub = df.loc[ix]
        if declared is not None and len(sub) != declared:
            raise AssertionError(f"N проб {len(sub)} ≠ объявленному {declared}")
        trials = sub["sr_d"].dropna().tolist()
        df.loc[ix, "n_prob"] = len(sub)
        df.loc[ix, "dsr"] = [dsr(r.sr_d, trials, int(r.T), r.skew, r.kurt)["dsr"] if r.sr_d == r.sr_d else np.nan
                             for r in sub.itertuples()]
        df.loc[ix, "holm_p_vs_A"] = _holm(sub["p_vs_A"].fillna(1.0).values)
    return df
