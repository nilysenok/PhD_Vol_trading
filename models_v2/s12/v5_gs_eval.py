"""V5 лист 1a, G12/G13: таблицы оценки GS1–GS5. Определения прежние: общие даты `S.common`, метрики `v4_b_stats.metr` (формулы `hp_hyb_out`), `crit_panel` = `hp_stats_sel.crit_panel`, DM/бутстрап `hp_stats_tst.pair_stats`, MCS `v4_b_stats.mcs`, Холм `stats_d.holm_n`.
«go» — три условия против порогов CHECKPOINT.json (планка включает K8). Окна — по году цели."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s5.d import stats_d as SD
from models_v2.s6.xgb.d_comp import crit_wf
from models_v2.s10 import hp_stats_sel as S, hp_stats_tst as TST
from models_v2.s11 import v4_b_stats as ST, v4_e_go as GO, v4_fin_data as FD
from models_v2.s12 import v5_0_crit as CR

GS = ["GS1", "GS2", "GS3", "GS4", "GS5"]
BASE = ["k1", "k2", "k3", "d2", "K8"]
COMPS = tuple(BASE)
WIN = {"2017": (2017, 2017), "2018_2021": (2018, 2021), "2022": (2022, 2022), "2023_2025": (2023, 2025), "2026": (2026, 2026), "2022_2025": (2022, 2025), "2017_2021": (2017, 2021), "all": (2017, 2026)}
TEST_WINS = {"2017_2021": "aux", "2022_2025": "main", "all": "main"}
SBER_REF = 0.2081


def wmask(k: pd.DataFrame, w: str) -> np.ndarray:
    lo, hi = WIN[w]
    y = k["t_plus"].dt.year
    return ((y >= lo) & (y <= hi)).to_numpy()


def full_cols(gs: list) -> list:
    """Те же ряды, что в листе 0 (4 контроля + 46 конфигураций V4) + GS: общие даты как в порогах CHECKPOINT."""
    return FD.CTRL + FD.CFGS + gs


def metrics(c: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Бумага × окно × ряд: QLIKE, RMSE, MAE, R² (ST.metr) + bias = mean ln(F/RV)."""
    rows = []
    for t, g in c.groupby("ticker"):
        for w in WIN:
            s = g[wmask(g, w)]
            for col in cols:
                if len(s):
                    rows.append({"ticker": t, "group": g["group"].iloc[0], "window": w, "series": col, **ST.metr(s["rv_tplus"], s[col]), "bias_ln": float(np.mean(np.log(s[col].to_numpy(float) / s["rv_tplus"].to_numpy(float))))})
    return pd.DataFrame(rows)


def comparison(c: pd.DataFrame, cols: list, pt: pd.DataFrame, cs: pd.Series, eqy: pd.Series) -> pd.DataFrame:
    m = pt[pt["group"] == "M16"].groupby(["window", "series"])[["qlike", "rmse", "mae", "r2_rv", "r2_ln", "bias_ln"]].mean().reset_index()
    rows = []
    for w in WIN:
        sub = m[m["window"] == w].set_index("series")
        for s in cols:
            d = {"d_qlike_%s_pct_of_k1" % r: 100.0 * float((sub.loc[s, "qlike"] - sub.loc[r, "qlike"]) / sub.loc["k1", "qlike"]) for r in COMPS}
            rows.append({"section": "metrics", "series": s, "window": w, "n_rows_m16": int(wmask(c[c["group"] == "M16"], w).sum()), **sub.loc[s].to_dict(), "crit_panel": float(cs[s]), "EqY": float(eqy[s]), **d})
    return pd.DataFrame(rows)


def go_table(k: pd.DataFrame, gs: list, thr: dict, eqy: pd.Series) -> pd.DataFrame:
    """Три условия «go» против порогов CHECKPOINT (cond1 — crit_panel, cond2 — без 2017, cond3 — без 1 % крайних), EqY рядом, решение по каждой GS."""
    cols = full_cols(gs)
    cp = S.crit_panel(k, cols).set_index("series")["crit_panel"]
    n17, tr = GO.crit_no2017(k, cols), GO.crit_trim(k, cols)
    rows = []
    for s in BASE + gs:
        c1, c2, c3 = cp[s] <= thr["cond1_crit_panel"], n17[s] <= thr["cond2_crit_no2017"], tr[s] <= thr["cond3_crit_trim1"]
        rows.append({"section": "go", "series": s, "crit_panel": float(cp[s]), "crit_no2017": float(n17[s]), "crit_trim1": float(tr[s]), "thr1": thr["cond1_crit_panel"], "thr2": thr["cond2_crit_no2017"], "thr3": thr["cond3_crit_trim1"],
                     "c1": bool(c1), "c2": bool(c2), "c3": bool(c3), "go": bool(c1 and c2 and c3) if s in gs else np.nan, "EqY": float(eqy[s])})
    return pd.DataFrame(rows)


def eqy_series(k: pd.DataFrame, cols: list) -> pd.Series:
    a = CR.annual(k, cols)
    return a.groupby(["series", "year"])["ann"].mean().groupby("series").mean()


def per_ticker(pt: pd.DataFrame, gs: list) -> pd.DataFrame:
    """17 бумаг: QLIKE и Δ к K1 в % уровня K1 по окнам; улучшение GS к K8 и к лучшему из K2/K3 (отдельные строки-счётчики внизу)."""
    t = pt[pt["series"].isin(BASE + gs) & pt["window"].isin(["2017_2021", "2022_2025", "all"])].copy()
    q = t.pivot_table(index=["ticker", "window"], columns="series", values="qlike")
    out = t.merge(q["k1"].rename("q_k1").reset_index(), on=["ticker", "window"])
    out["d_qlike_k1_pct_of_k1"] = 100.0 * (out["qlike"] - out["q_k1"]) / out["q_k1"]
    best23 = np.minimum(q["k2"], q["k3"])
    cnt = []
    for w in ("2017_2021", "2022_2025", "all"):
        for s in gs:
            qs = q.xs(w, level="window")
            cnt.append({"ticker": "COUNT", "window": w, "series": s, "n_improve_vs_K8": int((qs[s] < qs["K8"]).sum()), "n_improve_vs_best_k2k3": int((qs[s] < np.minimum(qs["k2"], qs["k3"])).sum()), "n_tickers": int(len(qs))})
    return pd.concat([out.drop(columns=["q_k1"]), pd.DataFrame(cnt)], ignore_index=True)


def years(c: pd.DataFrame, cols: list) -> pd.DataFrame:
    m = c[c["group"] == "M16"].copy()
    m["year"] = m["t_plus"].dt.year
    rows = []
    for s in cols:
        q = ST.qrow(m, s)
        g = pd.DataFrame({"ticker": m["ticker"].to_numpy(), "year": m["year"].to_numpy(), "q": q}).groupby(["ticker", "year"])["q"].mean().groupby("year").mean()
        rows += [{"series": s, "year": int(y), "qlike_m16": float(v)} for y, v in g.items()]
    return pd.DataFrame(rows)


def tests(c: pd.DataFrame, gs: list, n_decl: int, n_wide: int) -> pd.DataFrame:
    """GS × {K1, K2, K3, D2, K8}, M16, три окна: DM (НУ 5) + блочный бутстрап дат; p_holm (N = n_decl), p_holm_wide (N = n_wide) по основному семейству (2022–2025 и вся)."""
    rows = []
    for w, kind in TEST_WINS.items():
        tab = ST.date_table(c, COMPS + tuple(gs), w if w != "2017_2021" else "2017_2021")
        for s in gs:
            for comp in COMPS:
                rows.append({"config": s, "comparator": comp, "scope": "M16", "window": w, "kind": kind, **TST.pair_stats(tab[s].to_numpy() - tab[comp].to_numpy())})
    df = pd.DataFrame(rows)
    sel = (df["kind"] == "main") & df["dm_p"].notna()
    for name, n in (("p_holm", n_decl), ("p_holm_wide", n_wide)):
        df[name] = np.nan
        df.loc[sel, name] = SD.holm_n(df.loc[sel, "dm_p"].to_numpy(), n)
    df["n_main_family"] = int(sel.sum())
    return df


def crit_tk(k: pd.DataFrame, cols: list) -> pd.DataFrame:
    """crit_wf по бумагам M16 (те же строки, что `S.crit_panel`); среднее по бумагам = crit_panel."""
    w = S.crit_window(S.common(k, cols))
    out = {}
    for s in cols:
        r = w["rv_tplus"].to_numpy(float) / w[s].to_numpy(float)
        q = pd.Series(r - np.log(r) - 1.0, index=w.index)
        out[s] = {t: crit_wf(q[g.index].to_numpy(), g["t_plus"]) for t, g in w.groupby("ticker") if g["t_plus"].dt.year.nunique() >= 2}
    return pd.DataFrame(out)
