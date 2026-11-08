"""V5 лист 2: таблицы SP_PARAMS (θ по месяцам и знаки по полосам) и SP_ABLATION («спектр сверх формы дня»: SP2 против SP4 против H3 V4 против K8 на общих парах). Остальные таблицы — функции листа 1a (`v5_gs_eval*`, `v5_gs_enc`), общие и без правок.
Определения прежние: общие даты `S.common`, QLIKE M16 `v4_b_stats`, DM (НУ 5) и бутстрап дат `hp_stats_tst.pair_stats`. Диагностика, в выбор не входит."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S, hp_stats_tst as TST
from models_v2.s11 import v4_b_stats as ST, v4_e_go as GO
from models_v2.s12 import v5_gs_eval as EV

SP = ["SP1", "SP2", "SP3", "SP4"]
ABL = ["K8", "H3", "SP2", "SP4"]
PAIRS = [("SP4", "SP2"), ("SP2", "H3"), ("SP4", "H3"), ("SP2", "K8"), ("SP4", "K8"), ("H3", "K8")]
ABL_WINS = ("2017_2021", "2022_2025", "all")


def params_table(par: dict) -> pd.DataFrame:
    """Строки section=month: параметры по месяцам; section=summary: по параметру θ — медиана и доля месяцев с θ > 0 среди месяцев с оценкой (fitted)."""
    p = pd.concat([d.assign(config=g, section="month") for g, d in par.items()], ignore_index=True)
    rows = []
    for g, d in par.items():
        d = d[d["fitted"].astype(bool)] if "fitted" in d else d
        for c in [c for c in d.columns if c.startswith("th_")]:
            rows.append({"section": "summary", "config": g, "name": c, "median": float(d[c].median()), "share_pos": float((d[c] > 0).mean()), "n_months": int(len(d))})
    return pd.concat([p, pd.DataFrame(rows)], ignore_index=True)


def crit_all(k: pd.DataFrame, cols_full: list) -> pd.DataFrame:
    """crit_panel, без 2017, без 1 % для всех рядов кадра (в т. ч. H3 V4) — те же функции, что в `go_table`."""
    cp = S.crit_panel(k, cols_full).set_index("series")["crit_panel"]
    n17, tr = GO.crit_no2017(k, cols_full), GO.crit_trim(k, cols_full)
    return pd.DataFrame({"series": cols_full, "crit_panel": [float(cp[s]) for s in cols_full], "crit_no2017": [float(n17[s]) for s in cols_full], "crit_trim1": [float(tr[s]) for s in cols_full]})


def ablation(cc: pd.DataFrame, go: pd.DataFrame, eqy: pd.Series) -> pd.DataFrame:
    """Строки section=level: QLIKE M16 по окнам для K8, H3, SP2, SP4; section=crit: crit_panel / без 2017 / без 1 % / EqY; section=pair: Δ QLIKE (A − B) по датам, DM, бутстрап (A лучше B при Δ < 0)."""
    rows = []
    for w in ABL_WINS:
        tab = ST.date_table(cc, ABL, w)
        for s in ABL:
            rows.append({"section": "level", "window": w, "a": s, "qlike_m16_date_mean": float(tab[s].mean()), "n_dates": int(len(tab))})
        for a, b in PAIRS:
            rows.append({"section": "pair", "window": w, "a": a, "b": b, **TST.pair_stats(tab[a].to_numpy() - tab[b].to_numpy())})
    g = go.set_index("series")
    for s in ABL:
        rows.append({"section": "crit", "a": s, "crit_panel": float(g.loc[s, "crit_panel"]), "crit_no2017": float(g.loc[s, "crit_no2017"]), "crit_trim1": float(g.loc[s, "crit_trim1"]), "EqY": float(eqy[s])})
    for a, b in PAIRS:
        rows.append({"section": "crit_diff", "a": a, "b": b, "d_crit_panel": float(g.loc[a, "crit_panel"] - g.loc[b, "crit_panel"]), "d_crit_no2017": float(g.loc[a, "crit_no2017"] - g.loc[b, "crit_no2017"]),
                     "d_crit_trim1": float(g.loc[a, "crit_trim1"] - g.loc[b, "crit_trim1"]), "d_EqY": float(eqy[a] - eqy[b])})
    return pd.DataFrame(rows)
