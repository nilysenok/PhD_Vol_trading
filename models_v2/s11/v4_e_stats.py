"""V4-E, G24: статистика этапа E. Определения те же, что в B/C (`v4_b_stats`: общие даты, окна по году цели, M16, SBER отдельно); компараторы — K1, K2, K3, D2, B2, C8. DM — `hp_stats_tst.pair_stats`; Холм — `stats_d.holm_n`."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s5.d import stats_d as SD
from models_v2.s10 import hp_stats_tst as TST
from models_v2.s11 import v4_b_stats as ST

COMPS = ("k1", "k2", "k3", "d2", "B2", "C8")


def tests(c: pd.DataFrame, cfgs: list, n_decl: int, n_wide: int) -> pd.DataFrame:
    """E × компараторы на M16 × 4 окна; Δ = QLIKE(E) − QLIKE(база) по датам; p_holm (N = n_decl), p_holm_wide (N = n_wide); основное семейство M16 × {2022_2025, all}."""
    rows = []
    for w in ST.WINS:
        tab = ST.date_table(c, list(COMPS) + cfgs, w)
        for cfg in cfgs:
            for comp in COMPS:
                rows.append({"config": cfg, "comparator": comp, "scope": "M16", "window": w, "kind": "main" if w in ST.MAIN_WINS else "aux", **TST.pair_stats(tab[cfg].to_numpy() - tab[comp].to_numpy())})
    df = pd.DataFrame(rows)
    sel = (df["kind"] == "main") & df["dm_p"].notna()
    for name, n in (("p_holm", n_decl), ("p_holm_wide", n_wide)):
        df[name] = np.nan
        df.loc[sel, name] = SD.holm_n(df.loc[sel, "dm_p"].to_numpy(), n)
    df["n_main_family"] = int(sel.sum())
    return df


def bias_table(fc: dict) -> pd.DataFrame:
    """b по месяцам × бумагам и среднее по панели (PANEL) для каждой E; для E6 — и месячный вес w на M."""
    rows = []
    for cfg, f in fc.items():
        f = f.assign(month=f["date"].dt.to_period("M").astype(str))
        g = f.groupby(["month", "ticker"], as_index=False).agg(b=("b", "mean"), w=("w", "mean"), n=("b", "size"))
        p = g.groupby("month", as_index=False).agg(b=("b", "mean"), w=("w", "mean"), n=("n", "sum"))
        p["ticker"] = "PANEL"
        rows.append(pd.concat([g, p], ignore_index=True).assign(config=cfg))
    return pd.concat(rows, ignore_index=True)[["config", "month", "ticker", "b", "w", "n"]]


def shocks(fc: dict) -> pd.DataFrame:
    """Описательно: среднее по панели b в окнах шоков — март–апрель 2018 (санкции), 24.02–31.03.2022 — и в спокойное окно (2019) для сравнения."""
    wins = {"2018-03..04 (санкции)": ("2018-03-01", "2018-04-30"), "2022-02-24..03-31": ("2022-02-24", "2022-03-31"), "2019 (спокойно)": ("2019-01-01", "2019-12-31")}
    rows = []
    for cfg, f in fc.items():
        for name, (a, b) in wins.items():
            s = f[(f["date"] >= a) & (f["date"] <= b)]
            rows.append({"config": cfg, "window": name, "mean_b": float(s["b"].mean()) if len(s) else np.nan, "min_b": float(s["b"].min()) if len(s) else np.nan, "max_b": float(s["b"].max()) if len(s) else np.nan, "n": len(s)})
    return pd.DataFrame(rows)
