"""EX1 (R5, G4): информационная дополнительность без обучения бустингов. Пары прогнозов (корреляции, расхождения по RV/стрессу, доля «одна права»),
предсказуемость знака разности потерь (логит и дерево глубины 2, годовое переобучение по целям < 01.01.Y) и остатка HAR на F1–F6 (ridge, годовой LOYO)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from models_v2.s5 import pit
from models_v2.s6.e import comp, crit_e, pe, resid

PAIRS = (("HAR", "XJ"), ("HAR", "XF"), ("HAR", "LF"), ("XJ", "LF"), ("XF", "LF"), ("XJ", "XF"))
RIGHT = 0.3        # «права»: |ln RV − ln σ̂²| < 0.3 (объявлено до счёта)
YEARS = tuple(range(2018, 2027))


def _auc(y, p):
    return float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")


def sign_predictability(X, lab, td, yr) -> dict:
    """Знак (qA < qB) по признакам на d: обучение на строках с target < 01.01.Y, оценка на строках года Y (AUC)."""
    out = {"logit": [], "tree2": []}
    for Y in YEARS:
        tr, te = td < np.datetime64("%d-01-01" % Y), yr == Y
        if tr.sum() < 100 or te.sum() < 50:
            continue
        sc = StandardScaler().fit(X[tr])
        lg = LogisticRegression(C=0.1, max_iter=500).fit(sc.transform(X[tr]), lab[tr])
        tr2 = DecisionTreeClassifier(max_depth=2, min_samples_leaf=50, random_state=42).fit(X[tr], lab[tr])
        out["logit"].append(_auc(lab[te], lg.predict_proba(sc.transform(X[te]))[:, 1]))
        out["tree2"].append(_auc(lab[te], tr2.predict_proba(X[te])[:, 1]))
    return {k: float(np.nanmean(v)) if v else float("nan") for k, v in out.items()} | {"auc_logit_by_year": ";".join("%.3f" % a for a in out["logit"])}


def pair_rows(df, ter, X) -> pd.DataFrame:
    rv, td = df["rv"].to_numpy(float), df["target_date"].to_numpy("datetime64[D]")
    yr = pd.Series(df["target_date"]).dt.year.to_numpy()
    rows = []
    for a, b in PAIRS:
        sa, sb = comp.col(df, a), comp.col(df, b)
        ea, eb = np.log(rv / sa), np.log(rv / sb)
        qa, qb = crit_e.qlike_series(rv, sa), crit_e.qlike_series(rv, sb)
        r = {"pair": "%s|%s" % (a, b), "n": len(rv), "corr_ln_forecast": float(np.corrcoef(np.log(sa), np.log(sb))[0, 1]), "corr_ln_err": float(np.corrcoef(ea, eb)[0, 1]),
             "corr_qlike": float(np.corrcoef(qa, qb)[0, 1]), "mean_dq_a_minus_b": float((qa - qb).mean()), "sd_dq": float((qa - qb).std(ddof=1)),
             "share_a_better": float((qa < qb).mean()), "a_right_b_wrong": float(((abs(ea) < RIGHT) & (abs(eb) >= RIGHT)).mean()),
             "b_right_a_wrong": float(((abs(eb) < RIGHT) & (abs(ea) >= RIGHT)).mean()), "both_wrong": float(((abs(ea) >= RIGHT) & (abs(eb) >= RIGHT)).mean())}
        for lab, m in (("low_rv", ter == 0), ("mid_rv", ter == 1), ("high_rv", ter == 2), ("calm_g4", ~df["st_g4"].to_numpy(bool)), ("stress_g4", df["st_g4"].to_numpy(bool)),
                       ("stress_rv", df["st_rv"].to_numpy(bool))):
            r["dq_" + lab] = float((qa - qb)[m].mean()) if m.sum() > 20 else float("nan")
            r["corr_err_" + lab] = float(np.corrcoef(ea[m], eb[m])[0, 1]) if m.sum() > 20 else float("nan")
        r.update({"auc_" + k: v for k, v in sign_predictability(X, (qa < qb).astype(int), td, yr).items()})
        rows.append(r)
    return pd.DataFrame(rows)


def ridge_rows(df, G: dict, names: dict) -> pd.DataFrame:
    """ĝ на остатке e = ln RV − ln HAR-J(б): ridge (alpha = n_train) по группам признаков; годовой LOYO: обучение на целях < 01.01.Y, оценка в году Y."""
    e = np.log(df["rv"].to_numpy(float) / df["HAR"].to_numpy(float))
    td, yr, rv, h = df["target_date"].to_numpy("datetime64[D]"), pd.Series(df["target_date"]).dt.year.to_numpy(), df["rv"].to_numpy(float), df["HAR"].to_numpy(float)
    rows = []
    for nm, X in G.items():
        pred, s2h = np.full(len(e), np.nan), np.zeros(len(e))
        for Y in YEARS:
            tr, te = (td < np.datetime64("%d-01-01" % Y)), yr == Y
            if tr.sum() < 100 or te.sum() == 0:
                continue
            if X is None:
                pred[te], s2h[te] = e[tr].mean(), e[tr].var()
            else:
                sc = StandardScaler().fit(X[tr])
                mdl = Ridge(alpha=float(tr.sum())).fit(sc.transform(X[tr]), e[tr])
                pred[te], s2h[te] = mdl.predict(sc.transform(X[te])), float(((e[tr] - mdl.predict(sc.transform(X[tr]))) ** 2).mean())
        for Y in list(YEARS) + ["all"]:
            m = np.isfinite(pred) & (yr == Y if Y != "all" else True)
            if m.sum() < 50:
                continue
            q0, q1 = crit_e.qlike_series(rv[m], h[m]), crit_e.qlike_series(rv[m], h[m] * np.exp(pred[m] + s2h[m] / 2))
            rows.append({"model": nm, "n_features": 0 if X is None else X.shape[1], "year": Y, "n": int(m.sum()), "r2_vs_har": float(1 - ((e[m] - pred[m]) ** 2).sum() / (e[m] ** 2).sum()),
                         "r2_vs_mean": float(1 - ((e[m] - pred[m]) ** 2).sum() / ((e[m] - e[m].mean()) ** 2).sum()), "mean_e": float(e[m].mean()), "mean_ghat": float(pred[m].mean()),
                         "qlike_har": float(q0.mean()), "qlike_har_exp_g": float(q1.mean()), "dq_exp_g_minus_har": float((q1 - q0).mean())})
    return pd.DataFrame(rows)


def stage(_a=None) -> int:
    W = resid.build_world()
    df, g, F = comp.load(), W["grid"], W["F"]
    if not (df["d"].to_numpy("datetime64[ns]") == F.dates[g].astype("datetime64[ns]")).all():
        raise SystemExit("стоп: сетка comp != сетка мира")
    ter = np.nan_to_num(pit.past_bucket(np.exp(F.X[:, 1]), [1 / 3, 2 / 3]), nan=-1)[g]
    X = np.nan_to_num(W["dz"].X[g])
    feats = pd.read_parquet(resid.B2DIR / "feats_b2.parquet")
    import json
    acc = json.loads((resid.B2DIR / "accepted.json").read_text())
    G = {"const": None, **{k: np.nan_to_num(feats[v].to_numpy(float)[g]) for k, v in acc.items()}}
    G["F1-F6"] = np.nan_to_num(feats[[c for v in acc.values() for c in v]].to_numpy(float)[g])
    G["Б-2+F4"] = X
    pairs, res = pair_rows(df, ter, X), ridge_rows(df, G, acc)
    pe.write(pairs, pe.ENS / "INFO_PAIRS.csv", "EX1: пары прогнозов; HAR=HAR-J(б), XJ=x2_best_j, XF=ctrl_f4, LF=C0_inh_F4; терцили RV_w по прошлому; «права»: |ln-ошибка|<0,3")
    pe.write(res, pe.ENS / "HAR_RESIDUAL_PREDICTABILITY.csv", "EX1: остаток ln RV − ln HAR-J(б) на F1–F6, ridge годовой LOYO (alpha = n_train); const — только смещение; QLIKE-эффект: HAR·exp(ĝ + s²/2), s² — остаток обучения")
    a = res[(res.year == "all")].set_index("model")
    print("G4 EX1: R2_vs_mean (все годы):", a["r2_vs_mean"].round(4).to_dict(), "dQLIKE exp(g)-HAR:", a["dq_exp_g_minus_har"].round(5).to_dict(), flush=True)
    return 0
