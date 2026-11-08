"""V4-RQC, G36: статистика листа. Определения те же, что в B/C/E/CS/NTP/CLD (`v4_b_stats`: общие даты, окна по году цели, M16); компараторы — K1, K2, K3, D2, K8 (кроме самого K8).
DM — `hp_stats_tst.pair_stats`; Холм — `stats_d.holm_n`, основное семейство M16 × {2022_2025, all} × (K8, RQC1–RQC3) × компараторы. Таблица a — по месяцам × бумагам и по окнам шоков (`v4_e_stats.shocks`)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s5.d import stats_d as SD
from models_v2.s10 import hp_stats_tst as TST
from models_v2.s11 import v4_b_stats as ST, v4_e_stats as ES

COMPS = ("k1", "k2", "k3", "d2", "K8")


def tests(c: pd.DataFrame, cfgs: list, n_decl: int, n_wide: int) -> pd.DataFrame:
    """cfgs × компараторы на M16 × 4 окна; Δ = QLIKE(cfg) − QLIKE(база) по датам; p_holm (N = n_decl), p_holm_wide (N = n_wide)."""
    rows = []
    for w in ST.WINS:
        tab = ST.date_table(c, list(COMPS) + [x for x in cfgs if x not in COMPS], w)
        for cfg in cfgs:
            for comp in (x for x in COMPS if x != cfg):
                rows.append({"config": cfg, "comparator": comp, "scope": "M16", "window": w, "kind": "main" if w in ST.MAIN_WINS else "aux", **TST.pair_stats(tab[cfg].to_numpy() - tab[comp].to_numpy())})
    df = pd.DataFrame(rows)
    sel = (df["kind"] == "main") & df["dm_p"].notna()
    for name, n in (("p_holm", n_decl), ("p_holm_wide", n_wide)):
        df[name] = np.nan
        df.loc[sel, name] = SD.holm_n(df.loc[sel, "dm_p"].to_numpy(), n)
    df["n_main_family"] = int(sel.sum())
    return df


def a_table(fc: dict, fc_e: dict) -> pd.DataFrame:
    """ln a по месяцам × бумагам и среднее по панели (kind = month); окна шоков — RQC и E1–E5 (kind = window, ln a = mean_b, min/max). У E ln-множитель b — к F0 (H или B2), не к K8: сравнение поведения, не уровней."""
    lr = {c: f.assign(b=np.log(f["a"]), w=np.nan) for c, f in fc.items() if c != "K8"}
    m = ES.bias_table(lr).rename(columns={"b": "log_a"}).assign(kind="month")
    m["a"] = np.exp(m["log_a"])
    sh = ES.shocks({**lr, **fc_e}).rename(columns={"mean_b": "log_a", "min_b": "min_log_a", "max_b": "max_log_a"})
    sh = sh.rename(columns={"window": "month"}).assign(ticker="PANEL-окно", kind="window", a=lambda d: np.exp(d["log_a"]))
    return pd.concat([m, sh], ignore_index=True)[["config", "kind", "month", "ticker", "log_a", "a", "min_log_a", "max_log_a", "n"]]
