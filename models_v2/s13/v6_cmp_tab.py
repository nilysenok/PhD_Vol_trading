"""V6 лист Z2, G62/G63: таблицы сравнения SBER V1–V6 из СОХРАНЁННЫХ прогнозов (обучения нет). Статистика — те же реализации, что в V6 FIN (импорт, не копия): Δ = mean QLIKE(X) − mean QLIKE(Y) в % уровня K1 окна,
90 % интервал — блоковый бутстрап (блок 20, 2000 повторов, seed 0) `v6_1_eval.delta_ci`; DM — Ньюи–Уэст лаг 5, односторонний «X лучше» `v6_1_eval.dm_one`; Холм — `stats_d.holm_n` (N = 722 и N_wide = 3122, и N программы);
MCS 90 % — `stats_d.mcs` с `hp_stats_tst.block_idx`. Окна — по году цели t⁺ (`v6_0_q.WIN`)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_tst as TST
from models_v2.s5.d import stats_d as SD
from models_v2.s13 import v6_0_q as Q, v6_1_eval as EV, v6_cmp_reg as R

WINS = ("2017_2021", "2022_2025", "2026", "all")
BASES = ("K1", "K2", "K3", "K8")
N_MAIN, N_WIDE = 722, 3122
SET_MAIN, SET_NO6, SET_EXT, SET_ALL = "основное (K1, K2, K3, M, K8 + гибриды (а), (б))", "основное без V6", "расширенное (все PIT-ряды V1–V6, кроме A0g)", "все ряды таблицы (с потолками)"
YEARS = list(range(2017, 2027))


def labels(reg: pd.DataFrame) -> list:
    return list(dict.fromkeys(list(R.CTRL) + list(reg["series"])))


def win_q(u: dict, s: str, w: str) -> np.ndarray:
    m = Q.wmask(u["tp"], w)
    return Q.ql(u["rv"][m], u["f"][s][m])


def windows(u: dict, labs: list) -> pd.DataFrame:
    rows = []
    for s in labs:
        for w in WINS:
            q, k1, k2, k3 = (win_q(u, x, w).mean() for x in (s, "K1", "K2", "K3"))
            row = dict(series=s, window=w, n=int(Q.wmask(u["tp"], w).sum()), qlike=float(q), d_pct_K1=100 * float(q - k1) / float(k1), yrs_neg_K2=np.nan, yrs_neg_K3=np.nan, yrs_n=np.nan)
            if w in ("2017_2021", "2022_2025"):
                row.update(years_neg(u, s, w))
            rows.append(row)
    return pd.DataFrame(rows)


def years_neg(u: dict, s: str, w: str) -> dict:
    """Число лет окна, где средний QLIKE ряда < K2 (K3) того же года, и число лет окна."""
    m, lo, hi = Q.wmask(u["tp"], w), *Q.WIN[w]
    ys = list(range(lo, hi + 1))
    a, b, c = (Q.annual(win_q(u, x, w), u["tp"][m], ys) for x in (s, "K2", "K3"))
    return dict(yrs_neg_K2=int(sum(a[y] < b[y] for y in ys)), yrs_neg_K3=int(sum(a[y] < c[y] for y in ys)), yrs_n=len(ys))


def years(u: dict, labs: list) -> pd.DataFrame:
    rows = []
    for s in labs:
        for y in YEARS:
            m = (pd.DatetimeIndex(u["tp"]).year == y)
            rows.append(dict(series=s, year=y, n=int(m.sum()), qlike=float(Q.ql(u["rv"][m], u["f"][s][m]).mean())))
    return pd.DataFrame(rows)


def one_test(u: dict, s: str, b: str, w: str) -> dict:
    m = Q.wmask(u["tp"], w)
    rv, fx, fb = u["rv"][m], u["f"][s][m], u["f"][b][m]
    k1 = float(Q.ql(rv, u["f"]["K1"][m]).mean())
    return dict(**EV.delta_ci(Q.ql(rv, fx), Q.ql(rv, fb), k1), p_one=EV.dm_one(rv, fb, fx)["p_one"], n=int(m.sum()))


def tests(u: dict, reg: pd.DataFrame) -> pd.DataFrame:
    h = reg[reg["role"] != "контроль"]
    grp = h.groupby(["row", "series"], sort=False).agg(program=("program", "first"), role=("role", lambda x: "/".join(dict.fromkeys(x))), cls=("cls", "first")).reset_index()
    cache, rows = {}, []
    for _, g in grp.iterrows():
        for w in ("all", "2017_2021", "2022_2025"):
            for b in BASES:
                if g["series"] != b:
                    cache.setdefault((g["series"], b, w), one_test(u, g["series"], b, w))
                    rows.append(dict(row=g["row"], program=g["program"], series=g["series"], role=g["role"], cls=g["cls"], window=w, base=b, **cache[(g["series"], b, w)]))
    t = pd.DataFrame(rows)
    nprog = {sp["row"]: sp["n"] for sp in R.SPEC}
    t["N_prog"] = t["row"].map(nprog)
    for w in t["window"].unique():
        s = t["window"] == w
        t.loc[s, "p_holm_N722"] = SD.holm_n(t.loc[s, "p_one"].to_numpy(), N_MAIN)
        t.loc[s, "p_holm_N3122"] = SD.holm_n(t.loc[s, "p_one"].to_numpy(), N_WIDE)
        for r in t.loc[s, "row"].unique():
            g = s & (t["row"] == r)
            t.loc[g, "p_holm_Nprog"] = SD.holm_n(t.loc[g, "p_one"].to_numpy(), int(nprog[r]))
    return t


def mcs_one(u: dict, ser: list, w: str) -> pd.DataFrame:
    m = Q.wmask(u["tp"], w)
    L = np.column_stack([Q.ql(u["rv"][m], u["f"][s][m]) for s in ser])
    r = SD.mcs(L, TST.block_idx(len(L)))
    return pd.DataFrame([dict(window=w, model=s, mean_qlike=float(L[:, i].mean()), mcs_p=float(r["p"][i]), in_mcs90=bool(r["in"][i]), n_dates=len(L)) for i, s in enumerate(ser)])


def mcs(u: dict, reg: pd.DataFrame) -> pd.DataFrame:
    cand = R.candidates(reg)
    sets = {SET_MAIN: cand, SET_ALL: labels(reg)}
    out = [mcs_one(u, ser, w).assign(set=name) for name, ser in sets.items() for w in R.Q_WIN]
    t = pd.concat(out, ignore_index=True)
    t["cls"] = [R.klass(s)[0] for s in t["model"]]
    return t[["set", "window", "model", "cls", "mean_qlike", "mcs_p", "in_mcs90", "n_dates"]]
