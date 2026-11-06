"""Стадия inter (GATES_HF G5, HF8): тест взаимодействия «режим V1–V6 × относительная ошибка HAR/ML». Режим на d — только по прошлому (PIT-терцили, закрытые прогнозы).
dq_t = QLIKE(HAR) − QLIKE(ML) на цели d+1; OLS dq на индикаторы групп (константа + k−1), ковариация Ньюи–Уэст (лаг 5, Бартлетт), Вальд χ²(k−1); 6×2 тестов, Холм по 12."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from models_v2.s5 import pit
from models_v2.s5.d import ens
from models_v2.s7 import hf_data as D

LAG, ALPHA, VS = 5, 0.10, ("V1", "V2", "V3", "V4", "V5", "V6")
FEB24 = np.datetime64("2022-02-24")


def wald_nw(y, grp) -> tuple:
    """OLS y на константу + индикаторы групп (группа −1 отброшена); Вальд χ²(k−1) на равенство средних групп, ковариация НW (лаг 5). -> (stat, p, k, n)."""
    y, grp = np.asarray(y, float), np.asarray(grp)
    ok = (grp >= 0) & np.isfinite(y)
    y, grp = y[ok], grp[ok].astype(int)
    lv = sorted(set(grp.tolist()))
    if len(lv) < 2:
        return np.nan, np.nan, len(lv), len(y)
    X = np.column_stack([np.ones(len(y))] + [(grp == g).astype(float) for g in lv[1:]])
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    s = X * (y - X @ b)[:, None]
    S = s.T @ s
    for l in range(1, LAG + 1):
        G = s[l:].T @ s[:-l]
        S += (1 - l / (LAG + 1)) * (G + G.T)
    V = XtXi @ S @ XtXi
    Rb, RV = b[1:], V[1:, 1:]
    stat = float(Rb @ np.linalg.pinv(RV) @ Rb)
    return stat, float(sps.chi2.sf(stat, len(lv) - 1)), len(lv), len(y)


def regimes(w: dict, q_har: np.ndarray, q_ml: np.ndarray, rv, d, td) -> dict:
    """V1–V6 на строках сетки; -1 — нет истории. Пороги терцилей — по 252 предыдущим (`pit.past_bucket`), без текущего."""
    F, g, feats = w["F"], w["g"], w["feats"]
    bk = lambda x: np.nan_to_num(pit.past_bucket(x, [1 / 3, 2 / 3]), nan=-1)[g].astype(int)
    lw, lm = F.X[:, 1], F.X[:, 2]
    dlw = np.full(len(lw), np.nan)
    dlw[5:] = lw[5:] - lw[:-5]
    m = ens.closed({"rv": rv, "d": d, "td": td})
    cq = np.vstack([np.zeros((1, 2)), np.cumsum(np.column_stack([q_har, q_ml]), axis=0)])
    v6 = np.full(len(rv), -1)
    for k in range(len(rv)):
        if m[k] >= 63:
            v6[k] = int(cq[m[k], 0] - cq[m[k] - 63, 0] - (cq[m[k], 1] - cq[m[k] - 63, 1]) > 0)
    return {"V1": D.ter_w(F, g), "V2": bk(lw - lm), "V3": np.where(np.isnan(dlw[g]), -1, (dlw[g] > 0).astype(int)),
            "V4": (feats["f2_jfreq22"].to_numpy()[g] > 0).astype(int), "V5": bk(feats["f6_mkt_chg"].to_numpy(float)), "V6": v6}


def sign_diff(dq, grp, mask) -> float:
    """Среднее dq старшей группы минус младшей на подвыборке mask."""
    m = mask & (grp >= 0)
    if not m.any():
        return np.nan
    hi, lo = grp[m].max(), 0
    a, b = dq[m & (grp == hi)], dq[m & (grp == lo)]
    return float(a.mean() - b.mean()) if len(a) and len(b) else np.nan


def holm_sets(p):
    from models_v2.s3 import dm as dmod
    return dmod.holm(np.asarray(p, float))


def table(w: dict) -> pd.DataFrame:
    df = w["df"]
    rv, d, td = df["rv"].to_numpy(float), df["d"].to_numpy("datetime64[ns]"), df["target_date"].to_numpy("datetime64[ns]")
    yr, ex = D.year(df), td.astype("datetime64[D]") != FEB24
    qh, rows = D.qs(df, df["HAR"]), []
    for ml in ("XJ", "LF"):
        qm = D.qs(df, df[ml])
        dq = qh - qm
        for v, grp in regimes(w, qh, qm, rv, d, td).items():
            s, p, k, n = wald_nw(dq, grp)
            s2, p2, _k, n2 = wald_nw(dq[ex], grp[ex])
            means = [float(dq[grp == j].mean()) for j in range(k)] if k > 1 else []
            e, l = sign_diff(dq, grp, yr <= 2021), sign_diff(dq, grp, yr >= 2022)
            rows.append({"V": v, "ML": ml, "k": k, "n": n, "wald": s, "p": p, "n_ex0222": n2, "wald_ex0222": s2, "p_ex0222": p2,
                         "dq_mean_by_group": ";".join("%.4f" % x for x in means), "diff_hi_lo_2017_2021": e, "diff_hi_lo_2022_2026": l,
                         "sign_same": bool(np.isfinite(e) and np.isfinite(l) and e * l > 0)})
    t = pd.DataFrame(rows)
    t["p_holm"], t["p_holm_ex0222"] = holm_sets(t["p"].fillna(1.0)), holm_sets(t["p_ex0222"].fillna(1.0))
    t["confirmed"] = (t["p_holm"] < ALPHA) & (t["p_holm_ex0222"] < ALPHA) & t["sign_same"]
    return t


def stage(_a=None) -> int:
    w = D.world()
    t = table(w)
    D.write(t, "REGIME_INTERACTION.csv", "HF8: Вальд (НW лаг 5) взаимодействия режим × dq=QLIKE(HAR)−QLIKE(ML); dq>0 — ML лучше; Холм по 12; confirmed = Холм<0,10 на всём периоде и без 24.02.2022 и знак совпал в 2017–21/2022–26")
    ok = t[t["confirmed"]]
    verdict = ("ПОДТВЕРЖДЕНО: " + ",".join("%s/%s" % (a, b) for a, b in zip(ok["V"], ok["ML"]))) if len(ok) else "ABANDON HF-G8: взаимодействие режим × относительная ошибка не подтверждено (ни один из 12 тестов не прошёл Холм<0,10 с устойчивым знаком)"
    print("INTER:", verdict, flush=True)
    print(t[["V", "ML", "k", "n", "wald", "p", "p_holm", "p_holm_ex0222", "diff_hi_lo_2017_2021", "diff_hi_lo_2022_2026", "confirmed"]].round(4).to_string(index=False), flush=True)
    return 0
