"""V4-B, G12: сравнение, статистика, MCS. Все ряды — на ОДИНАКОВЫХ датах (S.common: конечные, > 0 у всех рядов и RV). Окна — по году цели. M16 — среднее по бумагам; SBER — отдельно (B_SBER), в выбор не входит.
Метрики: QLIKE (`models_v2.metrics`), rmse_ln / mae_ln (логарифмы, определение проекта `metrics.summary`), R² на уровне σ² (1 − ΣRSS/ΣTSS). Статистика — `hp_stats_tst`: pair_stats (DM НУ 5, блочный бутстрап дат), block_idx; Холм — `stats_d.holm_n`."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s5.d import stats_d as SD
from models_v2.s10 import hp_stats_sel as S, hp_stats_tst as TST
from models_v2.s11 import v4_io as IO

WINS = IO.WINDOWS
MAIN_WINS = ("2022_2025", "all")       # основное семейство Холма: два окна — две гипотезы; 2017–2021 (выбор) и 2026 (неполный) — описательно
COMPS = ("k1", "k2", "k3", "d2")


def win_mask(k: pd.DataFrame, w: str) -> np.ndarray:
    lo, hi = WINS[w]
    y = k["t_plus"].dt.year
    return ((y >= lo) & (y <= hi)).to_numpy()


def metr(rv, f) -> dict:
    rv, f = np.asarray(rv, float), np.asarray(f, float)
    s = metrics.summary(rv, f)
    tss = float(((rv - rv.mean()) ** 2).sum())
    return {"n": s["n"], "qlike": s["qlike"], "rmse_ln": s["rmse_ln"], "mae_ln": s["mae_ln"], "r2_lvl": 1.0 - float(((rv - f) ** 2).sum()) / tss if tss > 0 else np.nan}


def per_ticker(c: pd.DataFrame, cols: list) -> pd.DataFrame:
    rows = []
    for (t, g), w in ((x, w) for x in c.groupby("ticker") for w in WINS):
        s = g[win_mask(g, w)]
        for col in cols:
            if len(s):
                rows.append({"ticker": t, "group": g["group"].iloc[0], "window": w, "series": col, **metr(s["rv_tplus"], s[col])})
    return pd.DataFrame(rows)


def comparison(c: pd.DataFrame, cols: list, crit: pd.DataFrame, pt: pd.DataFrame) -> pd.DataFrame:
    """Ряд × окно: среднее по M16 каждой метрики (среднее по бумагам), crit_panel, Δ QLIKE к K1 в % от уровня K1."""
    m16 = pt[pt["group"] == "M16"].groupby(["window", "series"])[["qlike", "rmse_ln", "mae_ln", "r2_lvl"]].mean().reset_index()
    n = c[c["group"] == "M16"]
    rows = []
    for w in WINS:
        sub = m16[m16["window"] == w].set_index("series")
        n_rows = int(win_mask(n, w).sum())
        for col in cols:
            rows.append({"series": col, "window": w, "n_rows_m16": n_rows, **sub.loc[col].to_dict(), "crit_panel": float(crit.set_index("series").loc[col, "crit_panel"]),
                         "d_qlike_k1": float(sub.loc[col, "qlike"] - sub.loc["k1", "qlike"]), "d_qlike_k1_pct_of_k1": 100.0 * float((sub.loc[col, "qlike"] - sub.loc["k1", "qlike"]) / sub.loc["k1", "qlike"])})
    return pd.DataFrame(rows)


def qrow(c: pd.DataFrame, s: str) -> np.ndarray:
    r = c["rv_tplus"].to_numpy(float) / c[s].to_numpy(float)
    return r - np.log(r) - 1.0


def date_table(c: pd.DataFrame, cols: list, w: str, group: str = "M16") -> pd.DataFrame:
    """Панельный QLIKE по датам (среднее по бумагам группы дня); индекс — дата прогноза."""
    m = ((c["group"] == group).to_numpy() & win_mask(c, w))
    Q = pd.DataFrame({s: qrow(c, s) for s in cols})
    return Q[m].groupby(c["date"].to_numpy()[m]).mean().sort_index()


def tests(c: pd.DataFrame, cfgs: list, n_decl: int, n_wide: int) -> pd.DataFrame:
    """B × {K1, K2, K3, D2} на M16 × 4 окна; DM (НУ лаг 5) + бутстрап дат; p_holm (N = n_decl) и p_holm_wide (N = n_wide) по основному семейству."""
    rows = []
    for w in WINS:
        tab = date_table(c, list(COMPS) + cfgs, w)
        for cfg in cfgs:
            for comp in COMPS:
                rows.append({"config": cfg, "comparator": comp, "scope": "M16", "window": w, "kind": "main" if w in MAIN_WINS else "aux", **TST.pair_stats(tab[cfg].to_numpy() - tab[comp].to_numpy())})
    df = pd.DataFrame(rows)
    sel = (df["kind"] == "main") & df["dm_p"].notna()
    for name, n in (("p_holm", n_decl), ("p_holm_wide", n_wide)):
        df[name] = np.nan
        df.loc[sel, name] = SD.holm_n(df.loc[sel, "dm_p"].to_numpy(), n)
    df["n_main_family"] = int(sel.sum())
    return df


def mcs(c: pd.DataFrame, cols: list) -> pd.DataFrame:
    """MCS 90 % (T_max, блочный бутстрап дат) на панельном QLIKE M16 по датам; окна выбора, подтверждения и вся история."""
    rows = []
    for w in ("2017_2021", "2022_2025", "all"):
        tab = date_table(c, cols, w)
        r = SD.mcs(tab.to_numpy(), TST.block_idx(len(tab)))
        rows += [{"window": w, "model": m, "mean_qlike": float(tab[m].mean()), "mcs_p": float(r["p"][i]), "in_mcs90": bool(r["in"][i]), "n_dates": len(tab)} for i, m in enumerate(cols)]
    return pd.DataFrame(rows)


def sber(c: pd.DataFrame, cols: list, ref: float) -> pd.DataFrame:
    """SBER отдельно: метрики по окнам; ориентир 0,2081 (A1_roll_126, Ж-211) — потолок, сравнение на тех же датах окна all."""
    s = c[c["ticker"] == "SBER"]
    rows = []
    for w in WINS:
        sw = s[win_mask(s, w)]
        for col in cols:
            m = metr(sw["rv_tplus"], sw[col])
            rows.append({"series": col, "window": w, **m, "ref_a1_roll_126": ref if w == "all" else np.nan, "d_to_ref": m["qlike"] - ref if w == "all" else np.nan,
                         "note": "ориентир — потолок (c, λ выбраны на всей истории)" if w == "all" else ""})
    return pd.DataFrame(rows)
