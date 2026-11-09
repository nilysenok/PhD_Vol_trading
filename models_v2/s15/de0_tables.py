"""DE0: сборка таблиц G2 (разложение разрыва к K2) и G4 (устойчивость, предсказуемость) из массивов. Всё — чистые функции от (rv, f, year)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s15 import de0_core as C, de0_ridge as RG

HYB = ["A2", "NTP2", "cfg8", "cfg4"]
SER = ["K1", "K2"] + HYB
WIN2 = ("2018_2021", "2022_2025")


def l_of(rv, f: dict) -> dict:
    return {k: C.qlike_day(rv, v) for k, v in f.items()}


def gap_years(rv, f: dict, year) -> pd.DataFrame:
    l = l_of(rv, f)
    return pd.concat([C.contrib_years(l[h] - l["K2"], year).assign(hybrid=h) for h in HYB], ignore_index=True)[["hybrid", "year", "n", "mean_delta", "contrib"]]


def gap_base(rv, f: dict, year) -> pd.DataFrame:
    """(ii): Δ_H = QLIKE(H) − QLIKE(K2) и доля, закрытая гибридом: (QLIKE(H) − QLIKE(гибрид)) / Δ_H; >1 — гибрид лучше K2."""
    rows = []
    for h in HYB:
        for w in ("2018_2021", "2022_2025", "2018_2025"):
            q = {s: C.under_over(rv, f[s], year, w)["qlike"] for s in ("K1", "K2", h)}
            dH = q["K1"] - q["K2"]
            rows.append(dict(hybrid=h, window=w, n=int(C.wmask(year, w).sum()), q_H=q["K1"], q_K2=q["K2"], q_hybrid=q[h], delta_H=dH, gain_over_H=q["K1"] - q[h], share_closed=(q["K1"] - q[h]) / dH, delta_hybrid_vs_K2=q[h] - q["K2"]))
    return pd.DataFrame(rows)


def under_over_table(rv, f: dict, year) -> pd.DataFrame:
    rows = []
    for s in SER:
        for w in ("2018_2021", "2022_2025", "2018_2025"):
            r = C.under_over(rv, f[s], year, w)
            rows.append(dict(series="H" if s == "K1" else s, window=w, **r, identity=C.identity_under_over(r)))
    return pd.DataFrame(rows)


def tails_table(rv, f: dict, year) -> pd.DataFrame:
    l = l_of(rv, f)
    rows = []
    for h in HYB:
        for w in WIN2:
            m = C.wmask(year, w)
            for q in (0.01, 0.05):
                rows.append(dict(hybrid=h, window=w, **C.tail_share((l[h] - l["K2"])[m], q)))
    return pd.DataFrame(rows)


def causes_table(rv, f: dict, year) -> pd.DataFrame:
    return pd.DataFrame([C.cause_row(rv, f, year, h) for h in HYB])


def acf_table(rv, f: dict, year) -> pd.DataFrame:
    l = l_of(rv, f)
    rows = []
    for h in ("cfg8", "A2"):
        for w in WIN2:
            d = (l[h] - l["K2"])[C.wmask(year, w)]
            rows.append(dict(diff="l_%s − l_K2" % h, window=w, n=len(d), **{"lag%d" % k: v for k, v in C.acf(d).items()}, band95=1.96 / np.sqrt(len(d))))
    return pd.DataFrame(rows)


def runs_table(rv, f: dict, year) -> pd.DataFrame:
    l = l_of(rv, f)
    return pd.DataFrame([dict(diff="l_%s − l_K2" % h, window=w, **C.run_summary((l[h] - l["K2"])[C.wmask(year, w)])) for h in ("cfg8", "A2") for w in WIN2])


def ridge_table(rv, f: dict, year, dates, tp, feats: dict) -> pd.DataFrame:
    """feats: имя набора → X_of_year(Y). Цели: u_H = ln(RV/H), дифференциалы cfg8 и A2 к K2. -> по году t⁺ и сводная строка `пул` (все годы вне выборки)."""
    l = l_of(rv, f)
    tg = {"ln(RV/H)": C.u_of(rv, f["K1"]), "l_cfg8 − l_K2": l["cfg8"] - l["K2"], "l_A2 − l_K2": l["A2"] - l["K2"]}
    out = []
    for t, y in tg.items():
        for k, xf in feats.items():
            r = RG.pit_oos(xf, y, dates, tp)
            if r.empty:
                continue
            pool = dict(year="пул", n=int(r["n"].sum()), sse_model=float(r["sse_model"].sum()), sse_mean=float(r["sse_mean"].sum()))
            pool["r2_oos"] = 1.0 - pool["sse_model"] / pool["sse_mean"]
            out.append(pd.concat([r.assign(year=r["year"].astype(str)), pd.DataFrame([pool])]).assign(target=t, features=k))
    return pd.concat(out, ignore_index=True)[["target", "features", "year", "n", "sse_model", "sse_mean", "r2_oos"]]
