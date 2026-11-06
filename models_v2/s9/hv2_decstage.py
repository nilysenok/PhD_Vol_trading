"""Ж-214, лист 3, стадия `dec` (V7, ступени Л4, Л5): минимальный набор S1, S3 (знак), J1, J3 (скачок/непрерывная), T1, T3 (время суток) — 6 trials из 16 → V13 ПО СХЕМАМ (У15):
S — лучшая из S1, S3; J — J1 (J3 не даёт прогноза RV, отдельные метрики вероятности); T — лучшая из T1, T3; контроль — прямой A0 (D1). Остаток (S2, S4, J2, J4, T2, T4, плато) не считается (У15)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s9 import hv2_a0, hv2_dec as DC, hv2_eval as E, hv2_jlogit as JL, hv2_ledger as L, hv2_paths as P, hv2_world as W

DIR = "DEC"
MODS = (DC, JL, W, hv2_a0)
CTRL = ("A0", "XJ", "LF")
KIND = {"S1": "sign_own", "S3": "sign_shar_direct", "J1": "cj_own", "J3": "cj_logit", "T1": "tod_own", "T3": "tod_edges"}
SCHEME = {"S1": "S", "S3": "S", "J1": "J", "J3": "J", "T1": "T", "T3": "T"}
SUBDIR = {"S": "semivariance_jumps", "J": "semivariance_jumps", "T": "intraday_decomposition"}
DESC = {"S1": "RS+ и RS− — каждая log-HAR по своим d, w, m + ln(1+J); σ²=Σ", "S3": "прямая ln RV на [ln RS+_d, ln RS−_d, ln RV_w, ln RV_m, ln(1+J)] (SHAR-аналог, D1)",
        "J1": "C=RV−J: log-HAR по C + ln(1+J); J: ln(1+J) по своим d,w,m, Ĵ=max(expm1(f+s²/2),0); σ²=Ĉ+Ĵ", "J3": "P(J_{t+1}>0): логистика+ridge λ=1, прогноз будущих скачков",
        "T1": "OPEN, MIDDLE, CLOSE — каждая log-HAR по своим d,w,m + ln(1+J); σ²=Σ", "T3": "T1 + ln CLOSE_d в OPEN и MIDDLE, ln OPEN_d в CLOSE (края дня)"}


def register_all() -> dict:
    return {c: L.register("decomposition", {"name": c, "kind": KIND[c], "scheme": SCHEME[c], "spec": DESC[c], "controls": ["A0 (прямой D1)"]}) for c in KIND}


def compute(c: str, w: DC.DWorld, a0) -> dict:
    if c == "J3":
        ph, base, npair = JL.run(w)
        return {c: ph, c + "__base": base, c + "__npairs": npair.astype(float)}
    s2, comps, fb = DC.run(w, c, a0)
    return {c: s2, c + "__fb": fb.astype(float), **{"%s__c_%s" % (c, n): comps[n] for n in comps}}


def comp_table(c: str, ser: dict, w: DC.DWorld) -> pd.DataFrame:
    rows = []
    for n in DC.COMPS[c]:
        f = ser["%s__c_%s" % (c, n)]
        a = w.target if n == "rv" else w.comp[n][w.el + 1]
        ok = np.isfinite(f) & np.isfinite(a)
        pos = ok & (a > 0) & (f > 0)
        rows.append({"config": c, "component": n, "n": int(ok.sum()), "mean_forecast": float(f[ok].mean()), "mean_actual": float(a[ok].mean()), "bias_ratio": float(f[ok].mean() / a[ok].mean()),
                     "rmse": float(np.sqrt(np.mean((f[ok] - a[ok]) ** 2))), "mae": float(np.mean(np.abs(f[ok] - a[ok]))), "qlike_pos_days": float(np.mean(metrics.qlike(a[pos], f[pos]))) if pos.any() else np.nan,
                     "min_forecast": float(f[ok].min())})
    return pd.DataFrame(rows)


def by_year(df: pd.DataFrame, series: dict) -> pd.DataFrame:
    yr = pd.DatetimeIndex(df["target_date"]).year.to_numpy()
    q = {k: np.asarray(metrics.qlike(df["rv"], v), float) for k, v in series.items()}
    return pd.DataFrame([{"model": k, "year": int(Y), "n": int((yr == Y).sum()), "qlike": float(q[k][yr == Y].mean()), "qlike_A0": float(q["A0"][yr == Y].mean()),
                          "d_vs_A0": float((q[k] - q["A0"])[yr == Y].mean()), "d_vs_XJ": float((q[k] - q["XJ"])[yr == Y].mean())} for k in series if k != "A0" for Y in sorted(set(yr))])


def trunc(w: DC.DWorld, fn) -> float:
    return max(DC.trunc_diff(fn, w, int(w.el[k])) for k in E.TRUNC_AT)


def stage_dec(_a=None) -> int:
    ctx = E.context()
    w, a0 = DC.build(ctx["w"], pd.read_parquet(P.M10)), ctx["refs"]["A0"]
    keys = register_all()
    ser = {c: E.cached(DIR, c, MODS, lambda c=c: compute(c, w, a0), ctx) for c in KIND}
    rows, ident = [], []
    for c in KIND:
        if c == "J3":
            continue
        comps = {n: ser[c]["%s__c_%s" % (c, n)] for n in DC.COMPS[c]}
        fb = ser[c][c + "__fb"] > 0
        idn = DC.identities(ser[c][c], comps, fb)
        ident.append({"config": c, **idn})
        rows.append(E.row(ctx, c, ser[c][c], {"kind": KIND[c], "scheme": SCHEME[c], "key": keys[c], "trunc_max_diff": trunc(w, lambda x, c=c: DC.sigma(x, c)), **idn}, CTRL))
    refs = [E.row(ctx, k, v, {"kind": "контроль"}, CTRL) for k, v in ctx["refs"].items() if k in CTRL]
    for sch, name in (("S", "DECOMP_S.csv"), ("J", "DECOMP_J.csv"), ("T", "DECOMP_T.csv")):
        sub = [r for r in rows if r["scheme"] == sch]
        if sub:
            P.write_csv(pd.DataFrame(refs + sub), name, "V7, схема %s; Δ — строка минус контроль (A0 = прямой D1), минус — строка лучше; сумма компонент против прямого A0; потолок: best по crit_wf на всей истории" % sch, SUBDIR[sch])
    for sch in "SJT":
        cs = [c for c in KIND if SCHEME[c] == sch and c != "J3"]
        P.write_csv(pd.DataFrame([i for i in ident if i["config"] in cs]), "DEC_IDENTITIES_%s.csv" % sch, "тождества на прогнозах: min компонент (>0; для J >=0), Σ компонент = σ̂²_Σ, n_fallback", SUBDIR[sch])
        P.write_csv(pd.concat([comp_table(c, ser[c], w) for c in cs]), "DEC_COMPONENTS_%s.csv" % sch, "прогноз каждой компоненты против своего факта (уровни %²); QLIKE — на днях с фактом > 0", SUBDIR[sch])
        series = {"A0": ctx["refs"]["A0"], "XJ": ctx["refs"]["XJ"], **{c: ser[c][c] for c in cs}}
        P.write_csv(by_year(ctx["df"], series), "DEC_BY_YEAR_%s.csv" % sch, "стабильность по годам (год цели): QLIKE и Δ к A0, XJ", SUBDIR[sch])
    ph, base = ser["J3"]["J3"], ser["J3"]["J3__base"]
    jm = JL.metrics(w, ph, base)
    jm["trunc_max_diff"] = trunc(w, lambda x: JL.run(x)[0])
    P.write_csv(jm, "J3_FORECAST_FUTURE_JUMP.csv", "J3: ПРОГНОЗ P(J_{t+1}>0), AUC и Brier к базовой частоте по закрытым парам; auc_naive_persist — контроль 1[J_d>0]", SUBDIR["J"])
    P.write_csv(JL.diagnostic(pd.read_parquet(P.M10)), "J_DIAGNOSTIC_PAST_JUMPS.csv", "ДИАГНОСТИКА состоявшихся скачков (не прогноз и не вход моделей)", SUBDIR["J"])
    dec = {s: E.v13([r for r in rows if r["scheme"] == s], ("A0",), len(rows) + 1, L.LIMITS["decomposition"]) for s in "SJT"}
    yj = jm[~jm["period"].isin(["all", "ex_240222"])]
    j3 = {"auc_all": float(jm.loc[jm.period == "all", "auc"].iloc[0]), "brier_skill_all": float(jm.loc[jm.period == "all", "brier_skill"].iloc[0]),
          "years_auc_gt_half": int((yj["auc"] > 0.5).sum()), "years_skill_pos": int((yj["brier_skill"] > 0).sum()), "informative": bool(((yj["auc"] > 0.5) & (yj["brier_skill"] > 0)).sum() >= E.MIN_YEARS),
          "trunc_max_diff": float(jm["trunc_max_diff"].iloc[0])}
    E.decide("V7", {"by_scheme": dec, "passed_V13": [s for s in "SJT" if dec[s]["continue"]], "J3": j3, "minimal_set": list(KIND), "unused_trials": L.LIMITS["decomposition"] - len(KIND),
                    "not_run": ["S2", "S4", "J2", "J4", "T2", "T4", "плато"]})
    cols = ["model", "scheme", "qlike_all", "crit_wf", "d_crit_A0", "d_crit_XJ", "d_qlike_no_240222_A0", "years_better_A0", "n_fallback", "trunc_max_diff"]
    print(pd.DataFrame(rows)[cols].round(5).to_string(index=False))
    print(jm.round(4).to_string(index=False))
    print({s: (dec[s]["best"], dec[s]["continue"], dec[s]["controls"]) for s in "SJT"}, j3, flush=True)
    return 0 if all(r["trunc_max_diff"] == 0 for r in rows) and j3["trunc_max_diff"] == 0 else 1
