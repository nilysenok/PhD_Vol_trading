"""V6-1a, G14: модуль оценки конфигураций (общий для всех листов V6). Метрики — определение проекта (`v6_0_q`: QLIKE, RMSE, MAE, R², bias = mean ln(F/RV)); окна — по году цели t⁺ (годы 2017…2025, 2026, 2017_2021, 2022_2025, all).
Δ — в % среднего QLIKE K1 окна, интервал 90 % — блоковый бутстрап по датам (блок 20, 2000 повторов, seed 0). DM — Ньюи–Уэст лаг 5, односторонний p «модель лучше»; Холм — с N = 722 и N_wide = 3122 из CHECKPOINT ключ `0`
(не с числом строк таблицы, мутация № 103). «Go» — пять условий по порогам CHECKPOINT ключ `0`. «Новая информация» — план п. 2.2 (Δ < 0; DM односторонний p < 0,05; Δ < 0 не менее чем в 3 годах из 5; цели 2017–2021)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from models_v2.s3 import dm as dmod
from models_v2.s5.d import stats_d as SD
from models_v2.s13 import v6_0_q as Q

B, BLOCK, SEED = 2000, 20, 0
LEVELS = ("all", "2017_2021", "2022_2025")


def metrics_table(rv, series: dict, tp) -> pd.DataFrame:
    """Строка на (ряд, окно): n, qlike, rmse, mae, r2_rv, r2_ln, bias_ln. series: имя -> прогноз на тех же датах."""
    rows = []
    for name, f in series.items():
        for w in Q.WIN:
            m = Q.wmask(tp, w)
            if m.any():
                rows.append(dict(series=name, window=w, **Q.metr_row(np.asarray(rv)[m], np.asarray(f)[m])))
    return pd.DataFrame(rows)


def delta_ci(qx, qy, k1_level: float, b: int = B, block: int = BLOCK, seed: int = SEED) -> dict:
    """Δ = mean(qx − qy) в % уровня K1; 90 % интервал (5-й и 95-й перцентили) блокового бутстрапа средних разности."""
    d = np.asarray(qx, float) - np.asarray(qy, float)
    bm = SD.boot_means(d, SD.block_idx(len(d), B_=b, block=block, seed=seed))
    return dict(delta_pct=100 * float(d.mean()) / k1_level, lo90=100 * float(np.percentile(bm, 5)) / k1_level, hi90=100 * float(np.percentile(bm, 95)) / k1_level)


def dm_one(rv, f_base, f_model) -> dict:
    """DM (НУ, лаг 5): d = QLIKE(base) − QLIKE(model), плюс = модель лучше; p_one — односторонний «модель лучше»."""
    r = dmod.dm(np.asarray(rv, float), np.asarray(f_base, float), np.asarray(f_model, float))
    p1 = 0.5 * math.erfc(r["stat"] / math.sqrt(2)) if np.isfinite(r["stat"]) else float("nan")
    return dict(mean_d=r["mean_d"], stat=r["stat"], p_two=r["p"], p_one=p1, n=r["n"])


def holm_both(p, cp0: dict) -> tuple:
    """Холм с N и N_wide из CHECKPOINT ключ `0` (N = 722, N_wide = 3122); не с числом строк таблицы."""
    return SD.holm_n(p, int(cp0["N"])), SD.holm_n(p, int(cp0["N_wide"]))


def go_conditions(rv, f, tp, q_k2, q_k3, cp0: dict) -> pd.DataFrame:
    """Условия «go» 1–5 для ряда f на датах цели 2017–2021: значение, порог из CHECKPOINT, выполнено. Аргументы — массивы окна 2017–2021."""
    r = Q.crit_row(rv, f, tp, q_k2, q_k3)
    th = cp0["thresholds"]
    rows = [dict(cond=i + 1, name=nm, value=float(r[col]), threshold=float(th[key]["threshold"]), passed=bool(r[col] <= th[key]["threshold"]))
            for i, (nm, col, key) in enumerate([("crit_SBER", "crit", "cond1_crit"), ("без целей 2017", "crit_no2017", "cond2_crit_no2017"), ("без целей 2017–2018", "crit_no2017_18", "cond3_crit_no2017_18"),
                                                ("без 1 % крайних дат", "crit_trim1", "cond4_crit_trim1")])]
    yb = int(r["years_better_of5"])
    rows.append(dict(cond=5, name="лет лучше лучшего из K2/K3 (из 5)", value=float(yb), threshold=float(th["cond5_years_better_min"]), passed=bool(yb >= th["cond5_years_better_min"])))
    return pd.DataFrame(rows)


def new_information(rv, f_x, f_y, tp) -> dict:
    """План п. 2.2 на целях 2017–2021: (i) Δ = mean QLIKE(X) − mean QLIKE(Y) < 0; (ii) DM односторонний p < 0,05; (iii) Δ < 0 не менее чем в 3 годах из 5. Все три — «несёт новую информацию»."""
    m = Q.wmask(tp, "2017_2021")
    rv, fx, fy, t = np.asarray(rv, float)[m], np.asarray(f_x, float)[m], np.asarray(f_y, float)[m], np.asarray(tp)[m]
    qx, qy = Q.ql(rv, fx), Q.ql(rv, fy)
    ax, ay = Q.annual(qx, t), Q.annual(qy, t)
    neg = int(sum(ax[y] - ay[y] < 0 for y in ax if y in ay))
    p1 = dm_one(rv, fy, fx)["p_one"]
    i, ii, iii = bool(qx.mean() - qy.mean() < 0), bool(p1 < 0.05), bool(neg >= 3)
    return dict(delta=float(qx.mean() - qy.mean()), p_dm_one=p1, years_neg=neg, cond_i=i, cond_ii=ii, cond_iii=iii, new_information=bool(i and ii and iii))


def tests_table(rv, f, refs: dict, tp, cp0: dict, windows=LEVELS) -> pd.DataFrame:
    """A0 против контролей: Δ в % уровня K1 с интервалом, DM (p сырое односторонний), p Холма (N, N_wide) — по окнам. refs: имя -> прогноз; K1 обязателен (уровень)."""
    rows = []
    for w in windows:
        m = Q.wmask(tp, w)
        rvw, fw = np.asarray(rv)[m], np.asarray(f)[m]
        qx, k1 = Q.ql(rvw, fw), float(Q.ql(rvw, np.asarray(refs["K1"])[m]).mean())
        for name, base in refs.items():
            fb = np.asarray(base)[m]
            rows.append(dict(window=w, base=name, **delta_ci(qx, Q.ql(rvw, fb), k1), **dm_one(rvw, fb, fw)))
    t = pd.DataFrame(rows)
    t["p_holm_N"], t["p_holm_Nwide"] = np.nan, np.nan
    for w in windows:
        s = (t["window"] == w) & t["p_one"].notna()
        t.loc[s, "p_holm_N"], t.loc[s, "p_holm_Nwide"] = holm_both(t.loc[s, "p_one"].to_numpy(), cp0)
    return t
