"""V4-FIN: сводные таблицы по всем 46 конфигурациям. Определения — прежние (без новых): общие даты `S.common`, метрики `v4_b_stats.metr` (формулы `hp_hyb_out`), `crit_panel` = `hp_stats_sel.crit_panel`, «go» — `v4_e_go.rules` (три условия Ж-225 п. 10),
DM/бутстрап — `v4_b_stats.tests` (`hp_stats_tst.pair_stats`), MCS — `v4_b_stats.mcs`, годы — `v4_cb_years.years_table`. Новое здесь — только окна 2017 и 2018 отдельными строками и сборка в одну таблицу.
Три лучшие архитектуры — три конфигурации из 46 с наименьшим `crit_panel` (окно выбора, M16 × 2017–2021); K8 входит в 46 и в сравнение отдельно."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import v4_b_stats as ST, v4_cb_years as Y, v4_e_go as GO, v4_fin_data as FD

WINS6 = {"2017": (2017, 2017), "2018": (2018, 2018), "2017_2021": (2017, 2021), "2022_2025": (2022, 2025), "2026": (2026, 2026), "all": (2017, 2026)}
KEEP_T = ["k1", "k2", "k3", "d2", "K8", "C1", "C4"]
REFS = ("k2", "k3", "d2")


def _wm(k: pd.DataFrame, w: str) -> np.ndarray:
    lo, hi = WINS6[w]
    y = k["t_plus"].dt.year
    return ((y >= lo) & (y <= hi)).to_numpy()


def per_ticker(c: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Бумага × окно × ряд: n, qlike, rmse, mae, r2_rv, r2_ln (`ST.metr`, как в таблицах этапов; окна — 6)."""
    rows = []
    for t, g in c.groupby("ticker"):
        for w in WINS6:
            s = g[_wm(g, w)]
            rows += [{"ticker": t, "group": g["group"].iloc[0], "window": w, "series": col, **ST.metr(s["rv_tplus"], s[col])} for col in cols if len(s)]
    return pd.DataFrame(rows)


def best3(cs: pd.Series) -> list:
    return sorted(FD.CFGS, key=lambda x: (float(cs[x]), x))[:3]


def go_rules(k: pd.DataFrame, cs: pd.Series, thr: float) -> dict:
    return GO.rules(k, ["k1"] + FD.CFGS, cs, thr)


def conditions(g: dict, cols: list) -> pd.DataFrame:
    """Условия 1–3 и «go» по рядам; для K2, K3, D2 — только значения (crit без 2017, без 1 %): они задают планку условий 2 и 3."""
    rows = []
    for s in cols:
        r = g["cfg"].get(s)
        if r is None:
            r = {"c1": np.nan, "c2": np.nan, "c3": np.nan, "go": np.nan, **g["ref"][s]}
        rows.append({"series": s, **{x: r[x] for x in ("crit_no2017", "crit_trim1", "c1", "c2", "c3", "go")}})
    return pd.DataFrame(rows)


def all_table(c: pd.DataFrame, cols: list, cs: pd.Series, pt: pd.DataFrame, cond: pd.DataFrame) -> pd.DataFrame:
    """FIN_ALL: ряд × окно — среднее по M16 каждой метрики, Δ QLIKE к K1, K2, K3, D2 в % уровня K1 (того же окна), crit_panel, условия «go», простое среднее QLIKE 2017–2021."""
    m = pt[pt["group"] == "M16"].groupby(["window", "series"])[["qlike", "rmse", "mae", "r2_rv", "r2_ln"]].mean().reset_index()
    simple = m[m["window"] == "2017_2021"].set_index("series")["qlike"]
    cd = cond.set_index("series")
    rows = []
    for w in WINS6:
        sub = m[m["window"] == w].set_index("series")
        n = int(_wm(c[c["group"] == "M16"], w).sum())
        for s in cols:
            q1 = sub.loc["k1", "qlike"]
            d = {"d_qlike_%s_pct_of_k1" % r: 100.0 * float((sub.loc[s, "qlike"] - sub.loc[r, "qlike"]) / q1) for r in ("k1",) + REFS}
            rows.append({"series": s, "stage": FD.STAGE_OF.get(s, "контроль"), "window": w, "n_rows_m16": n, **sub.loc[s].to_dict(), "crit_panel": float(cs[s]), "qlike_simple_2017_2021": float(simple[s]), **d, **cd.loc[s].to_dict()})
    return pd.DataFrame(rows)


def tickers_table(pt: pd.DataFrame, top: list) -> pd.DataFrame:
    keep = KEEP_T + [x for x in top if x not in KEEP_T]
    t = pt[pt["series"].isin(keep)].copy()
    q1 = t[t["series"] == "k1"].set_index(["ticker", "window"])["qlike"]
    t["d_qlike_k1_pct_of_k1"] = [100.0 * (q - q1[(a, w)]) / q1[(a, w)] for q, a, w in zip(t["qlike"], t["ticker"], t["window"])]
    return t


def tests_table(c: pd.DataFrame, top: list, cp: dict) -> pd.DataFrame:
    cfgs = list(top) + [x for x in ("K8",) if x not in top]
    return ST.tests(c, cfgs, cp["N_declared_before_stats"], cp["N_wide"])


def mcs_table(c: pd.DataFrame, cols: list) -> pd.DataFrame:
    return ST.mcs(c, cols)


def years_table(k: pd.DataFrame, cols: list) -> pd.DataFrame:
    return Y.years_table(k, cols)


def crit_series(k: pd.DataFrame, cols: list) -> pd.Series:
    return S.crit_panel(k, cols).set_index("series")["crit_panel"]
