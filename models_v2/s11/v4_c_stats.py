"""V4-C, G17: статистика листа C-a. Те же определения, что в B (`v4_b_stats`: окна по году цели, общие даты, M16 — среднее по бумагам, SBER отдельно), компараторы расширены B6.
DM — `hp_stats_tst.pair_stats` (НУ лаг 5, блочный бутстрап по датам); Холм — `stats_d.holm_n`, основное семейство M16 × {2022_2025, all}; N = 694 (p_holm) и N_wide = 3094 (p_holm_wide)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s5.d import stats_d as SD
from models_v2.s10 import hp_stats_tst as TST
from models_v2.s11 import v4_b_stats as ST

COMPS = ("k1", "k2", "k3", "d2", "B6")


def tests(c: pd.DataFrame, cfgs: list, n_decl: int, n_wide: int) -> pd.DataFrame:
    """C × {K1, K2, K3, D2, B6} на M16 × 4 окна; Δ = QLIKE(C) − QLIKE(база) по датам; p_holm (N = n_decl), p_holm_wide (N = n_wide)."""
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


def block_steps(crit: pd.Series) -> pd.DataFrame:
    """Δ crit_panel по блокам признаков: C1 к постоянному весу (B2 нет в таблице — к K1), C2 − C1, C3 − C2, C4 − C3."""
    rows = [{"step": "C1 − K1 (z_div)", "d_crit": crit["C1"] - crit["k1"]}, {"step": "C2 − C1 (+ z_perf)", "d_crit": crit["C2"] - crit["C1"]},
            {"step": "C3 − C2 (+ z_mkt)", "d_crit": crit["C3"] - crit["C2"]}, {"step": "C4 − C3 (+ z_new: HAR-surprise, Intraday Shape)", "d_crit": crit["C4"] - crit["C3"]}]
    return pd.DataFrame(rows).assign(d_crit_pct_of_k1=lambda d: 100.0 * d["d_crit"] / crit["k1"])


def weight_by_year(fc: pd.DataFrame, col: str = "w") -> pd.Series:
    """Средний вес на M по году даты прогноза (M16 и 17 бумаг — среднее по строкам панели)."""
    return fc.groupby(fc["date"].dt.year)[col].mean()
