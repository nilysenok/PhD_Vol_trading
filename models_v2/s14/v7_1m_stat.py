"""V7-1a-M, статистика сравнения (план 6.1, 6.2, 6.4). Сравнение ряда F с K1, K2, K3, M = √(K2·K3), K8 = √(K1·M), V4:NTP2, V6:A2 на ОДИНАКОВЫХ датах (дата, где нет любого из рядов, выпадает для всех).
Δ = mean QLIKE(F) − mean QLIKE(база) в % среднего QLIKE K1 окна; 90 %-й интервал — блоковый бутстрап (блок 20, 2000 повторов, seed 0); DM — Ньюи–Уэст, лаг 5, односторонний p «F лучше»;
Холм с N = 733 и N_wide = 3166 (не с числом строк таблицы, мутация № 103); MCS на уровне 90 % (T_max, те же блоки). Реализации — импорт из V6 (`v6_1_eval`, `stats_d`); окна по году цели t⁺ (разработка 2018–2021)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s5.d import stats_d as SD
from models_v2.s13 import v6_0_q as Q, v6_1_eval as EV

N_HOLM, N_WIDE = 733, 3166
B, BLOCK, SEED = EV.B, EV.BLOCK, EV.SEED
assert (B, BLOCK, SEED) == (2000, 20, 0)
WIN = {"2018_2021": (2018, 2021), "2019_2021": (2019, 2021), "2022_2025": (2022, 2025), "2026": (2026, 2026), "all": (2018, 2026), **{str(y): (y, y) for y in range(2018, 2026)}}
BASES = ("K1", "K2", "K3", "M", "K8", "V4:NTP2", "V6:A2")


def refs(k1, k2, k3, ntp2, a2) -> dict:
    k1, k2, k3 = (np.asarray(x, float) for x in (k1, k2, k3))
    m = np.sqrt(k2 * k3)
    return {"K1": k1, "K2": k2, "K3": k3, "M": m, "K8": np.sqrt(k1 * m), "V4:NTP2": np.asarray(ntp2, float), "V6:A2": np.asarray(a2, float)}


def wmask(tp, w: str) -> np.ndarray:
    y = pd.DatetimeIndex(tp).year.to_numpy()
    lo, hi = WIN[w]
    return (y >= lo) & (y <= hi)


def common(rv, f, rf: dict, tp) -> tuple:
    """Только даты, где конечны и положительны rv, F и ВСЕ базы -> (rv, f, refs, tp)."""
    ok = np.isfinite(np.asarray(rv, float)) & (np.asarray(f, float) > 0)
    for v in rf.values():
        ok &= np.isfinite(v) & (v > 0)
    return np.asarray(rv, float)[ok], np.asarray(f, float)[ok], {k: np.asarray(v, float)[ok] for k, v in rf.items()}, np.asarray(tp)[ok]


def compare(rv, f, rf: dict, tp, windows=("2018_2021", "2019_2021", "2022_2025", "2026", "all")) -> pd.DataFrame:
    """Строка на (окно, база): n, Δ % уровня K1, 90 %-й интервал, DM (stat, p двусторонний, p односторонний), p Холма N и N_wide (по окну, по всем базам окна)."""
    rv, f, rf, tp = common(rv, f, rf, tp)
    rows = []
    for w in windows:
        m = wmask(tp, w)
        if not m.any():
            continue
        qf, k1 = Q.ql(rv[m], f[m]), float(Q.ql(rv[m], rf["K1"][m]).mean())
        for name in BASES:
            fb = rf[name][m]
            rows.append(dict(window=w, base=name, **EV.delta_ci(qf, Q.ql(rv[m], fb), k1, b=B, block=BLOCK, seed=SEED), **EV.dm_one(rv[m], fb, f[m])))
    t = pd.DataFrame(rows)
    t["p_holm_N"], t["p_holm_Nwide"] = np.nan, np.nan
    for w in t["window"].unique():
        s = (t["window"] == w) & t["p_one"].notna()
        t.loc[s, "p_holm_N"], t.loc[s, "p_holm_Nwide"] = EV.holm_both(t.loc[s, "p_one"].to_numpy(), {"N": N_HOLM, "N_wide": N_WIDE})
    return t


def mcs90(rv, series: dict, tp, window: str = "2018_2021") -> pd.DataFrame:
    """MCS 90 % (T_max) на потерях QLIKE окна, одинаковые даты у всех рядов. -> имя, mean_qlike, p, in_mcs."""
    names = list(series)
    ok = np.isfinite(np.asarray(rv, float)) & wmask(tp, window)
    for v in series.values():
        ok &= np.isfinite(v) & (np.asarray(v, float) > 0)
    L = np.column_stack([Q.ql(np.asarray(rv, float)[ok], np.asarray(series[k], float)[ok]) for k in names])
    res = SD.mcs(L, SD.block_idx(len(L), B_=B, block=BLOCK, seed=SEED), alpha=0.10)
    return pd.DataFrame({"series": names, "mean_qlike": L.mean(axis=0), "p": res["p"], "in_mcs": res["in"]})


def years_table(rv, f, rf: dict, tp, years=range(2018, 2026)) -> pd.DataFrame:
    """Δ QLIKE (F − база) по годам цели для каждой базы: знак по годам (правило 6.2: Δ < 0 в ≥ 3 годах из 4)."""
    rv, f, rf, tp = common(rv, f, rf, tp)
    y, qf = pd.DatetimeIndex(tp).year.to_numpy(), Q.ql(rv, f)
    return pd.DataFrame([dict(year=yr, base=b, delta=float(qf[y == yr].mean() - Q.ql(rv[y == yr], rf[b][y == yr]).mean())) for yr in years if (y == yr).any() for b in BASES])


def new_information(rv, f_x, f_y, tp) -> dict:
    """План 6.2, цели 2018–2021: Δ < 0; DM односторонний p < 0,05; Δ < 0 не менее чем в 3 годах из 4."""
    tp = np.asarray(tp)
    m = wmask(tp, "2018_2021") & np.isfinite(f_x) & np.isfinite(f_y)
    rv, fx, fy = (np.asarray(a, float)[m] for a in (rv, f_x, f_y))
    t = pd.DatetimeIndex(tp[m])
    qx, qy = Q.ql(rv, fx), Q.ql(rv, fy)
    ax, ay = Q.annual(qx, t, range(2018, 2022)), Q.annual(qy, t, range(2018, 2022))
    neg = int(sum(ax[y] - ay[y] < 0 for y in ax if y in ay))
    p1 = EV.dm_one(rv, fy, fx)["p_one"]
    ok = bool(qx.mean() - qy.mean() < 0 and p1 < 0.05 and neg >= 3)
    return dict(delta=float(qx.mean() - qy.mean()), p_dm_one=p1, years_neg=neg, new_information=ok)
