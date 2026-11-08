"""V4-CLD, G33: статистика листа. Определения те же, что в B/C/E/CS/NTP (`v4_b_stats`: общие даты, окна по году цели, M16); компараторы — K1, K2, K3, D2, C1, C4, C8, NTP1.
DM — `hp_stats_tst.pair_stats`; Холм — `stats_d.holm_n`, основное семейство M16 × {2022_2025, all} × CLD1–CLD4."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s5.d import stats_d as SD
from models_v2.s10 import hp_stats_tst as TST
from models_v2.s11 import v4_b_stats as ST

COMPS = ("k1", "k2", "k3", "d2", "C1", "C4", "C8", "NTP1")


def tests(c: pd.DataFrame, cfgs: list, n_decl: int, n_wide: int) -> pd.DataFrame:
    """CLD × компараторы на M16 × 4 окна; Δ = QLIKE(CLD) − QLIKE(база) по датам; p_holm (N = n_decl), p_holm_wide (N = n_wide)."""
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
