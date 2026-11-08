"""V5 лист FIN, G23: FIN_JOINT (совместные ошибки), FIN_ENCOMPASS (охват лучших GS и SP), FIN_ABLATION (цепочки абляции). Диагностика задним числом, в выбор не входит.
Ошибка e = ln(RV/F) на общих строках M16 (цель τ = следующий день); окна — по году цели. Охват — функции листа 1a (`v5_gs_enc.encompass`: Ньютон, круговой блочный бутстрап дат B = 500, блок 20, seed 42), без правок."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_tst as TST
from models_v2.s12 import v5_fin_data as D, v5_fin_tabs as TB, v5_gs_enc as EN, v5_gs_eval as EV, v5_sp_eval as SE

WINS = ("2017_2021", "2022_2025", "all")
CHAIN = [("K8", "k1", "H→K8"), ("GS2", "K8", "K8→GS2"), ("GS3", "GS2", "GS2→GS3"), ("GS4", "GS2", "GS2→GS4"), ("GS5", "GS2", "GS2→GS5"), ("SP1", "K8", "K8→SP1"), ("SP2", "K8", "K8→SP2"),
         ("SP3", "SP2", "SP2→SP3"), ("SP4", "SP2", "SP2→SP4"), ("SP2", "H3", "H3→SP2"), ("SP4", "H3", "H3→SP4"), ("H3", "K8", "K8→H3")]


def joint_series(d: dict) -> list:
    return ["k1", "k2", "k3", "K8", d["best_gs"], d["best_sp"]]


def cross_ticker_corr(m: pd.DataFrame, e: pd.Series) -> float:
    """Средняя попарная корреляция ошибок разных бумаг по общим датам (недиагональные элементы матрицы бумага × бумага; минимум 60 общих дат на пару)."""
    c = e.groupby([m["date"].to_numpy(), m["ticker"].to_numpy()]).mean().unstack().corr(min_periods=60).to_numpy()
    off = c[~np.eye(len(c), dtype=bool)]
    off = off[np.isfinite(off)]
    return float(off.mean()) if len(off) else float("nan")


def joint(d: dict) -> pd.DataFrame:
    J, cc, rows = joint_series(d), d["cc"], []
    for w in WINS:
        m = cc[(cc["group"] == "M16") & EV.wmask(cc, w)]
        e = pd.DataFrame({s: np.log(m["rv_tplus"].to_numpy(float) / m[s].to_numpy(float)) for s in J}, index=m.index)
        c, n = e.corr(), len(m)
        for a, b in itertools.combinations(J, 2):
            rows.append(dict(section="corr", window=w, a=a, b=b, value=float(c.loc[a, b]), n=n))
            rows.append(dict(section="same_sign", window=w, a=a, b=b, value=float((np.sign(e[a]) == np.sign(e[b])).mean()), n=n))
        sg = np.sign(e.to_numpy())
        rv, lo, hi = m["rv_tplus"].to_numpy(float), m[J].min(axis=1).to_numpy(float), m[J].max(axis=1).to_numpy(float)
        rows += [dict(section="all_same_sign", window=w, a="+".join(J), b="", value=float((np.abs(sg.sum(axis=1)) == len(J)).mean()), n=n),
                 dict(section="rv_below_min", window=w, a="+".join(J), b="", value=float((rv < lo).mean()), n=n), dict(section="rv_above_max", window=w, a="+".join(J), b="", value=float((rv > hi).mean()), n=n),
                 dict(section="rv_outside_range", window=w, a="+".join(J), b="", value=float(((rv < lo) | (rv > hi)).mean()), n=n)]
        rows += [dict(section="cross_ticker_corr", window=w, a=s, b="", value=cross_ticker_corr(m, e[s]), n=n) for s in J]
    return pd.DataFrame(rows)


def encompass(d: dict) -> pd.DataFrame:
    return pd.concat([EN.encompass(d["cc"], d["best_gs"]), EN.encompass(d["cc"], d["best_sp"])], ignore_index=True)


def ablation(d: dict) -> pd.DataFrame:
    """Строки section=crit: crit_panel / без 2017 / без 1 % / EqY ряда; section=pair: цепочка, Δ crit_panel, без 2017, без 1 %, EqY (a − b; минус = a лучше); section=qlike: Δ среднего панельного QLIKE M16 по датам и DM/бутстрап."""
    cc = d["cc"]
    cols = sorted({x for a, b, _ in CHAIN for x in (a, b)})
    ca = SE.crit_all(d["kk"], TB.COLS_CRIT).set_index("series")
    eqy = TB.crit_inputs(d)[1]
    rows = [dict(section="crit", a=s, crit_panel=float(ca.loc[s, "crit_panel"]), crit_no2017=float(ca.loc[s, "crit_no2017"]), crit_trim1=float(ca.loc[s, "crit_trim1"]), EqY=float(eqy[s])) for s in cols]
    tabs = {w: EV.ST.date_table(cc, cols, w) for w in WINS}
    for a, b, ch in CHAIN:
        rows.append(dict(section="pair", chain=ch, a=a, b=b, d_crit_panel=float(ca.loc[a, "crit_panel"] - ca.loc[b, "crit_panel"]), d_crit_no2017=float(ca.loc[a, "crit_no2017"] - ca.loc[b, "crit_no2017"]),
                         d_crit_trim1=float(ca.loc[a, "crit_trim1"] - ca.loc[b, "crit_trim1"]), d_EqY=float(eqy[a] - eqy[b])))
        rows += [dict(section="qlike", chain=ch, a=a, b=b, window=w, **TST.pair_stats(tabs[w][a].to_numpy() - tabs[w][b].to_numpy())) for w in WINS]
    return pd.DataFrame(rows)
