"""Ж-211: строка сравнения ряда σ̂² с опорами (HAR-J, S*): crit_wf, QLIKE по срезам, DM к опорам, годы с выигрышем, проверка обрезки на новых строках."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s6.e import comp, crit_e
from models_v2.s7 import hf_data as D


def refs_of(df: pd.DataFrame) -> dict:
    return {"harj": comp.col(df, "HAR"), "xj": comp.col(df, "XJ"), "lf": comp.col(df, "LF")}


def star(df: pd.DataFrame) -> tuple:
    """S* = лучший самостоятельный бустинг по crit_wf (HF1)."""
    c = {k: crit_e.cw(D.qs(df, comp.col(df, k)), df["target_date"]) for k in ("XJ", "LF")}
    k = min(c, key=lambda x: (c[x], x))
    return k, c


def years_won(df: pd.DataFrame, s2, ref) -> int:
    y, d = D.year(df), D.qs(df, s2) - D.qs(df, ref)
    return int(sum(d[y == k].mean() < 0 for k in range(2017, 2027)))


def row(df: pd.DataFrame, name: str, s2, **extra) -> dict:
    """Строка таблицы: crit_wf и срезы (`crit_e.describe`), Δ crit к HAR-J и к S*, годы с выигрышем к HAR-J / XJ / LF."""
    refs = refs_of(df)
    r = crit_e.describe(name, s2, df, refs)
    sk, cs = star(df)
    ch = crit_e.cw(D.qs(df, refs["harj"]), df["target_date"])
    r.update({"crit_har": ch, "dcrit_vs_har": r["crit_wf"] - ch, "star": sk, "dcrit_vs_star": r["crit_wf"] - cs[sk],
              "years_won_vs_har": years_won(df, s2, refs["harj"]), "years_won_vs_xj": years_won(df, s2, refs["xj"]),
              "years_won_vs_lf": years_won(df, s2, refs["lf"])})
    r.update(extra)
    return r


def truncation_max_diff(fn, rv, har, g, d, td, cuts=(300, 700, 1200, 1800)) -> float:
    """Главный тест обрезки: fn(rv, har, g, d, td) -> σ̂² (n,). Порча rv/har/g во всех строках j > K не меняет прогнозы строк k <= K. Возврат — max |Δ|."""
    base, worst = fn(rv, har, g, d, td), 0.0
    for K in cuts:
        rv2, har2, g2 = rv.copy(), har.copy(), g.copy()
        rv2[K + 1:] *= 37.0
        har2[K + 1:] *= 0.3
        g2[K + 1:] = -g2[K + 1:] + 1.7
        bad = fn(rv2, har2, g2, d, td)
        worst = max(worst, float(np.max(np.abs(bad[:K + 1] - base[:K + 1]))))
    return worst
