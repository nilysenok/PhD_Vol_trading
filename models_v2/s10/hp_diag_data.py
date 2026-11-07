"""P3c, G13: входы диагностики R15 — ТОЛЬКО обучение на 17 (`*_p17`, Ж-220) и готовые K2/K3 (`k23_<T>`); результаты на 28 (`hyb_xgb.parquet`, `loto.parquet`, `statistics/`) не читаются (мутация: проверка имён).
Совокупности: M16, SBER (обучение на 17), E11 — перенос (E11 в выводы по 17 не входят; строки E11 — отдельные, population = E11)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from models_v2.s10 import hp_paths as P

KEY = ["ticker", "date"]
MODS = ("xgb", "lgbm")
WINDOWS = ("all", "sel", "conf")
POPS = ("M16", "SBER", "E11")
OK_NAME = re.compile(r"(_p17[_.])|(^k23_[A-Z]+\.parquet$)")


def input_names() -> list:
    return ["hyb_xgb_p17.parquet", "hyb_lgbm_p17.parquet", "k45_p17.parquet", "loto16_p17_xgb.parquet", "loto16_p17_lgbm.parquet"] + ["k23_%s.parquet" % t for t in P.MAIN17 + ["GCHE"]]


def check_names(names) -> None:
    """Отказ, если вход не из обучения на 17 (результаты на 28 читать нельзя)."""
    bad = [n for n in names if not OK_NAME.search(n)]
    if bad:
        raise SystemExit("диагностика 17 читает только *_p17 и k23_<T>: %s" % bad)


def population(df: pd.DataFrame, pop: str) -> pd.DataFrame:
    """Строки совокупности по группе; E11 в 17 не попадает ни при каком pop из {M16, SBER}."""
    return df[df["group"] == pop]


def window(df: pd.DataFrame, w: str) -> pd.DataFrame:
    y, t = df["t_plus"].dt.year, df["t_plus"]
    m = {"all": t >= pd.Timestamp(P.FIRST_TARGET), "sel": (y >= 2017) & (y <= 2021), "conf": t >= pd.Timestamp("2022-01-01")}[w]
    return df[m]


def sub(df: pd.DataFrame, pop: str, w: str, cols: list) -> pd.DataFrame:
    """Строки совокупности и окна, где RV > 0 и все ряды cols конечны и > 0; порядок — дата, бумага."""
    d = window(population(df, pop), w)
    ok = np.isfinite(d["rv"]) & (d["rv"] > 0)
    for c in cols:
        ok &= np.isfinite(d[c]) & (d[c] > 0)
    return d[ok].sort_values(["date", "ticker"]).reset_index(drop=True)


def load() -> pd.DataFrame:
    """Одна таблица (бумага, дата): rv, k1, k4, k5 (+ число пар обучения), K2, K3 (где есть), для каждого модуля A1/B1/A2/D2 и поправка A1 (g, c, λ)."""
    check_names(input_names())
    k = pd.read_parquet(P.OUT / "k45_p17.parquet").rename(columns={"rv_tplus": "rv", "sigma2_k1": "k1", "sigma2_k4": "k4", "sigma2_k5": "k5", "k4_ntrain": "ntr_xgb", "k5_ntrain": "ntr_lgbm"})
    k = k[KEY + ["t_plus", "group", "gap", "rv", "k1", "k4", "k5", "ntr_xgb", "ntr_lgbm"]]
    for m in MODS:
        h = pd.read_parquet(P.OUT / ("hyb_%s_p17.parquet" % m))
        cols = {"sigma2_" + n: "%s_%s" % (m, n) for n in ("A1", "B1", "A2", "D2")}
        cols.update({"a1_g": m + "_g", "a1_c": m + "_c", "a1_lam": m + "_lam"})
        k = k.merge(h[KEY + list(cols)].rename(columns=cols), on=KEY, how="left")
    k23 = [pd.read_parquet(P.OUT / ("k23_%s.parquet" % t))[KEY + ["sigma2_k2", "sigma2_k3"]] for t in P.MAIN17 + ["GCHE"]]
    return k.merge(pd.concat(k23, ignore_index=True).rename(columns={"sigma2_k2": "k2", "sigma2_k3": "k3"}), on=KEY, how="left")


def load_loto() -> pd.DataFrame:
    """LOTO с поперечными признаками по 16 (У-25); прежний `loto_p17_*` в диагностику не идёт."""
    return pd.concat([pd.read_parquet(P.OUT / ("loto16_p17_%s.parquet" % m))[KEY + ["config", "sigma2"]] for m in MODS], ignore_index=True)


def row(cause: str, pop: str, w: str, item: str, metric: str, value, n=None, verdict: str = "") -> dict:
    return {"cause": cause, "population": pop, "window": w, "item": item, "metric": metric, "value": value, "n": n, "verdict": verdict}
