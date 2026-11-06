"""Ж-214, лист 3, стадия `dec2` (остаток V7, схема T; У22): T2, T4, P1, P2 — 4 trials (decomposition 6 → 10 из 16; P3, P4 не тратятся). Контроль — прямой A0 (D1) и T3/T1.
V13 к A0 по каждой строке и по схеме T в целом (лучшая из T1…T4, P1, P2). R2T на этих конфигурациях не пересчитывается (У17: компонентная база R2T — T1/T3)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s9 import hv2_a0, hv2_dec as DC, hv2_dec2 as D2, hv2_decstage as DS, hv2_eval as E, hv2_ledger as L, hv2_paths as P, hv2_qlog, hv2_world as W

DIR = "DEC2"
MODS = (D2, DC, hv2_qlog, W, hv2_a0)
CTRL = ("A0", "XJ", "LF")
KIND = {"T2": "tod_cross", "T4": "tod_qlike", "P1": "tod_edges_w", "P2": "tod_edges_qlike"}
DESC = {"T2": "OPEN/MIDDLE/CLOSE: свои d,w,m + ln(1+J) + ln d-лаг двух других частей, OLS в логарифмах", "T4": "признаки T1, QLIKE-подгонка по каждой части, σ̂²=exp(Xγ)",
        "P1": "T3 с краевой частью по w-среднему (плато T3)", "P2": "признаки T3, QLIKE-подгонка (плато T3)"}


def compute(c: str, w: DC.DWorld, a0) -> dict:
    s2, comps, fb = D2.run(w, c, a0)
    return {c: s2, c + "__fb": fb.astype(float), **{"%s__c_%s" % (c, n): comps[n] for n in comps}}


def comp_rows(c: str, ser: dict, w: DC.DWorld) -> pd.DataFrame:
    rows = []
    for n in D2.PARTS:
        f, a = ser["%s__c_%s" % (c, n)], w.comp[n][w.el + 1]
        ok = np.isfinite(f) & np.isfinite(a)
        pos = ok & (a > 0) & (f > 0)
        rows.append({"config": c, "component": n, "n": int(ok.sum()), "mean_forecast": float(f[ok].mean()), "mean_actual": float(a[ok].mean()), "bias_ratio": float(f[ok].mean() / a[ok].mean()),
                     "rmse": float(np.sqrt(np.mean((f[ok] - a[ok]) ** 2))), "mae": float(np.mean(np.abs(f[ok] - a[ok]))), "qlike_pos_days": float(np.mean(metrics.qlike(a[pos], f[pos]))), "min_forecast": float(f[ok].min())})
    return pd.DataFrame(rows)


def stage_dec2(_a=None) -> int:
    ctx = E.context()
    w, a0 = DC.build(ctx["w"], pd.read_parquet(P.M10)), ctx["refs"]["A0"]
    keys = {c: L.register("decomposition", {"name": c, "kind": KIND[c], "scheme": "T", "spec": DESC[c], "controls": ["A0 (прямой D1)"]}) for c in KIND}
    ser = {c: E.cached(DIR, c, MODS, lambda c=c: compute(c, w, a0), ctx) for c in KIND}
    old = pd.read_parquet(P.OUT / "DEC_series.parquet")
    ctx["refs"]["T3"] = old["T3"].to_numpy(float)
    rows, ident = [], []
    for c in KIND:
        comps = {n: ser[c]["%s__c_%s" % (c, n)] for n in D2.PARTS}
        idn = DC.identities(ser[c][c], comps, ser[c][c + "__fb"] > 0)
        ident.append({"config": c, **idn})
        tr = max(DC.trunc_diff(lambda x, c=c: D2.sigma(x, c), w, int(w.el[k])) for k in E.TRUNC_AT)
        rows.append(E.row(ctx, c, ser[c][c], {"kind": KIND[c], "scheme": "T", "key": keys[c], "trunc_max_diff": tr, **idn}, CTRL + ("T3",)))
    refs = [E.row(ctx, k, v, {"kind": "контроль"}, CTRL) for k, v in ctx["refs"].items() if k in CTRL]
    P.write_csv(pd.DataFrame(refs + rows), "DECOMP_T_REST.csv", "V7 остаток, схема T (У22); Δ — строка минус контроль (A0 = прямой D1; _T3 — минус T3), минус — строка лучше; потолок: best по crit_wf на всей истории", "intraday_decomposition")
    P.write_csv(pd.DataFrame(ident), "DEC_IDENTITIES_T_REST.csv", "тождества на прогнозах: min компонент (>0), Σ компонент = σ̂²_Σ, n_fallback", "intraday_decomposition")
    P.write_csv(pd.concat([comp_rows(c, ser[c], w) for c in KIND]), "DEC_COMPONENTS_T_REST.csv", "прогноз каждой части против факта (уровни %²)", "intraday_decomposition")
    series = {"A0": ctx["refs"]["A0"], "XJ": ctx["refs"]["XJ"], **{c: ser[c][c] for c in KIND}}
    P.write_csv(DS.by_year(ctx["df"], series), "DEC_BY_YEAR_T_REST.csv", "стабильность по годам (год цели): QLIKE и Δ к A0, XJ", "intraday_decomposition")
    allT = rows + [E.row(ctx, c, old[c].to_numpy(float), {"scheme": "T"}, ("A0",)) for c in ("T1", "T3")]
    dec = E.v13(allT, ("A0",), 10, L.LIMITS["decomposition"])
    each = {r["model"]: {"d_crit_A0": r["d_crit_A0"], "years_better_A0": r["years_better_A0"], "d_crit_T3": r.get("d_crit_T3"), "d_qlike_no_240222_A0": r["d_qlike_no_240222_A0"]} for r in rows}
    E.decide("V7_rest_T", {"scheme_T_all": dec, "each": each, "configs": list(KIND), "unused_trials": L.LIMITS["decomposition"] - 10, "not_run": ["P3", "P4", "S2", "S4", "J2", "J4"],
                           "note": "S, J не прошли V13 (V7) — их остаток не считается; R2T не пересчитывается"})
    cols = ["model", "qlike_all", "crit_wf", "d_crit_A0", "d_crit_XJ", "d_qlike_A0", "d_qlike_XJ", "d_qlike_no_240222_A0", "years_better_A0", "d_crit_T3", "n_fallback", "trunc_max_diff"]
    print(pd.DataFrame(rows)[cols].round(5).to_string(index=False))
    print("V13 схема T (все):", dec["best"], dec["continue"], dec["controls"], flush=True)
    return 0 if all(r["trunc_max_diff"] == 0 for r in rows) else 1
