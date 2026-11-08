"""V5 лист 1a, G13: стресс-проверки (без 2017; без 1 % крайних; только после 2021; без самой влиятельной бумаги), цена ограничения score (GS1 против GS2), LOTO, параметры θ по месяцам и сводка связей.
Без переоценки: меры пересчитываются по готовым прогнозам."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import v4_b_stats as ST, v4_e_go as GO
from models_v2.s12 import v5_gs_eval as EV

TRIM = 0.01


def influential(ct: pd.DataFrame, best: str) -> tuple:
    """Бумага M16 с наибольшим |вкладом| в Δ crit_panel лучшей GS к K8 (вклад = (crit_wf GS − crit_wf K8)/число бумаг)."""
    d = (ct[best] - ct["K8"]) / len(ct)
    return str(d.abs().idxmax()), d


def stress(k: pd.DataFrame, c: pd.DataFrame, gs: list, best: str, cols_full: list) -> pd.DataFrame:
    cp = S.crit_panel(k, cols_full).set_index("series")["crit_panel"]
    n17, tr = GO.crit_no2017(k, cols_full), GO.crit_trim(k, cols_full)
    ct = EV.crit_tk(k, cols_full)
    top, contrib = influential(ct, best)
    wo = ct.drop(index=top).mean()
    post = c[(c["group"] == "M16") & EV.wmask(c, "2022_2025")]
    pq = {s: float(pd.DataFrame({"t": post["ticker"], "q": ST.qrow(post, s)}).groupby("t")["q"].mean().mean()) for s in EV.BASE + gs}
    rows = []
    for s in EV.BASE + gs:
        for name, v in (("full_crit_panel", cp[s]), ("no_2017_crit_panel", n17[s]), ("no_1pct_extreme_crit_panel", tr[s]), ("only_after_2021_mean_qlike_2022_2025", pq[s]), ("without_top_ticker_crit_panel", wo[s])):
            ref = {"full_crit_panel": cp, "no_2017_crit_panel": n17, "no_1pct_extreme_crit_panel": tr, "only_after_2021_mean_qlike_2022_2025": pd.Series(pq), "without_top_ticker_crit_panel": wo}[name]
            rows.append({"stress": name, "series": s, "value": float(v), "delta_to_K8": float(v - ref["K8"]), "top_ticker": top if name.startswith("without") else "", "best_gs": best})
    return pd.DataFrame(rows)


def contributions(ct: pd.DataFrame, best: str) -> pd.DataFrame:
    top, d = influential(ct, best)
    return pd.DataFrame({"ticker": d.index, "contrib_to_delta_crit_vs_K8": d.to_numpy(), "is_top": [t == top for t in d.index]})


def robust(c: pd.DataFrame) -> pd.DataFrame:
    """GS1 (сырой) против GS2 (ограниченный): QLIKE M16 на 1 % строк с наибольшим RV/K8 («крайние дни», где ограничение работает) и на остальных; окна 2017–2021 и вся история."""
    rows = []
    for w in ("2017_2021", "all"):
        m = c[(c["group"] == "M16") & EV.wmask(c, w)]
        ratio = (m["rv_tplus"] / m["K8"]).to_numpy()
        ext = ratio >= np.sort(ratio)[::-1][max(int(round(TRIM * len(m))) - 1, 0)]
        for name, msk in (("extreme_1pct_rv_over_K8", ext), ("rest", ~ext), ("all", np.ones(len(m), bool))):
            sub = m[msk]
            q1, q2, q8 = (float(ST.qrow(sub, s).mean()) for s in ("GS1", "GS2", "K8"))
            rows.append({"window": w, "rows": name, "n": int(msk.sum()), "qlike_GS1": q1, "qlike_GS2": q2, "qlike_K8": q8, "GS2_minus_GS1": q2 - q1, "GS1_minus_K8": q1 - q8, "GS2_minus_K8": q2 - q8})
    return pd.DataFrame(rows)


def loto_table(k: pd.DataFrame, loto: dict, main: dict, cols_full: list) -> pd.DataFrame:
    """Для каждой GS и каждой бумаги M16: crit_wf бумаги в LOTO (модель её не видела), в основной модели и у K8. Итог — crit_panel LOTO (среднее по 16) и число бумаг, где LOTO лучше K8."""
    ct = EV.crit_tk(k, cols_full)
    rows = []
    for g, lk in loto.items():
        kk = k.merge(lk[["ticker", "date", "F"]].rename(columns={"F": g + "_loto"}), on=["ticker", "date"], how="left")
        cl = EV.crit_tk(kk, cols_full + [g + "_loto"])[g + "_loto"]
        for t in cl.index:
            rows.append({"config": g, "excluded": t, "crit_wf_loto": float(cl[t]), "crit_wf_main": float(ct.loc[t, g]), "crit_wf_K8": float(ct.loc[t, "K8"]), "loto_minus_K8": float(cl[t] - ct.loc[t, "K8"])})
        rows.append({"config": g, "excluded": "MEAN_M16", "crit_wf_loto": float(cl.mean()), "crit_wf_main": float(ct[g].mean()), "crit_wf_K8": float(ct["K8"].mean()), "loto_minus_K8": float(cl.mean() - ct["K8"].mean())})
    return pd.DataFrame(rows)


def params_tables(par: dict) -> tuple:
    """GS_PARAMS (θ по месяцам) и GS_W (по конфигурации и году: средние θ, доля месяцев с оценкой, равновесные строки, сводка связей)."""
    p = pd.concat([d.assign(config=g) for g, d in par.items()], ignore_index=True)
    p["year"] = p["month"].str[:4].astype(int)
    w = p.groupby(["config", "year"]).agg(rho=("rho", "mean"), alpha=("alpha", "mean"), beta=("beta", "mean"), share_fitted=("fitted", "mean"), n_equal_rows_mean=("n_equal_rows", "mean"), w_summary_mean=("w_summary", "mean")).reset_index()
    return p.drop(columns=["year"]), w
