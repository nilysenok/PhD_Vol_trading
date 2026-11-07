"""P3c, G13: R15 (а) охват, (б) сходство с бустингом, (в) индивидуальные эффекты. Каждая функция -> (строки CAUSES, метрики M16 для вывода, [вспомогательная таблица])."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s10 import hp_diag_core as C, hp_diag_data as D

CTRL = {"xgb": ("k4", "k2"), "lgbm": ("k5", "k3")}      # самостоятельный панельный бустинг без HAR-J и замороженный самостоятельный бустинг бумаги


def cause_a(df: pd.DataFrame) -> tuple:
    rows, met, aux = [], {}, []
    for pop in D.POPS:
        for w in D.WINDOWS:
            for kb in ("k2", "k3"):
                d = D.sub(df, pop, w, ["k1", kb])
                if len(d) < 250:
                    continue
                one = d["ticker"].nunique() == 1
                r = C.gamma_ticker(d["rv"], d["k1"], d[kb]) if one else C.gamma_panel(d["rv"], d["k1"], d[kb], d["date"])
                rows += [D.row("a", pop, w, kb, m, r[m], r["n"]) for m in ("gamma", "t", "p")]
                if pop == "M16":
                    met["t_%s_%s" % (kb, w)] = r["t"]
                if w in ("all", "conf") and not one:
                    for t, g in d.groupby("ticker"):
                        if len(g) >= 250:
                            aux.append({"ticker": t, "group": pop, "window": w, "base": kb, **C.gamma_ticker(g["rv"], g["k1"], g[kb])})
    return rows, met, pd.DataFrame(aux)      # SBER и GCHE — одна бумага: γ_ticker в основных строках


def cause_b(df: pd.DataFrame) -> tuple:
    rows, met = [], {}
    for pop in D.POPS:
        for w in ("all", "conf"):
            for m, (ctl, kb) in CTRL.items():
                d = D.sub(df, pop, w, ["k1", ctl, kb, m + "_A1", m + "_D2"])
                if len(d) < 100:
                    continue
                x, y = np.log(d[m + "_A1"]) - np.log(d[ctl]), np.log(d[kb]) - np.log(d["k1"])
                e_h, e_k = np.log(d["rv"]) - np.log(d[m + "_D2"]), np.log(d["rv"]) - np.log(d[kb])
                cd, ce = C.corr(x, y), C.corr(e_h, e_k)
                rows += [D.row("b", pop, w, m, "corr_delta_A1ctl_vs_kb_k1", cd, len(d)), D.row("b", pop, w, m, "corr_err_D2_vs_kb", ce, len(d))]
                if pop == "M16" and w == "all":
                    met["cd_" + m], met["ce_" + m] = cd, ce
    return rows, met, None


def cause_c(df: pd.DataFrame) -> tuple:
    rows, met = [], {}
    for pop in ("M16", "E11"):
        for w in D.WINDOWS:
            for m in D.MODS:
                d = D.sub(df, pop, w, ["k1", m + "_A1", m + "_B1"])
                if len(d) < 250:
                    continue
                ic = C.icc1(np.log(d["rv"]) - np.log(d[m + "_A1"]), d["ticker"])
                dq = metrics.qlike(d["rv"], d[m + "_B1"]) - metrics.qlike(d["rv"], d[m + "_A1"])
                per = pd.Series(np.asarray(dq)).groupby(d["ticker"]).mean()
                dt = C.date_mean_t(dq, d["date"])
                rows += [D.row("c", pop, w, m, "icc1", ic["icc"], len(d)), D.row("c", pop, w, m, "eta2", ic["eta2"], len(d)),
                         D.row("c", pop, w, m, "dQLIKE_B1_minus_A1", float(np.mean(dq)), len(d)), D.row("c", pop, w, m, "frac_tickers_B1_better", float((per < 0).mean()), len(per)),
                         D.row("c", pop, w, m, "t_dates_B1_minus_A1", dt["t"], dt["n_dates"])]
                if pop == "M16" and w == "all":
                    met["icc_" + m], met["dB_" + m] = ic["icc"], float(np.mean(dq))
    return rows, met, None
