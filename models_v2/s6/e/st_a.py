"""Тип A (R6, G5): комбинации прогнозов A1 HAR+X, A2 HAR+L, A3 X+L, A4 HAR+X+L; 5 правил × 4 = 20 trials + roll_126/roll_504 на лучшей комбинации (+2) = 22.
Контроли (средние, медианы, ctrl_f4-версии) — не trials. Веса по датам — CSV. X = x2_best_j (XJ), L = C0_inh_F4 (LF)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from models_v2.s5.c import mix
from models_v2.s6.e import comp, crit_e, pe, rules, ser

COMBOS = {"A1": ("HAR", "XJ"), "A2": ("HAR", "LF"), "A3": ("XJ", "LF"), "A4": ("HAR", "XJ", "LF")}
DIRS = {"A1": pe.REP / "har_xgboost", "A2": pe.REP / "har_lightgbm", "A3": pe.ENS, "A4": pe.REP / "har_xgb_lgbm"}
TRIAL_RULES = ("fixed_loyo", "inv_qlike_252", "inv_rmse_252", "roll_252", "shrink_252")


def refs_of(df) -> dict:
    return {"harj": comp.col(df, "HAR"), "b2x": comp.col(df, "B2X"), "b2l": comp.col(df, "B2L"), "xj": comp.col(df, "XJ"), "lf": comp.col(df, "LF")}


def controls(df, ctx) -> dict:
    """{id: (σ̂², члены)}: средние по уровню/логарифму и медиана для A1…A4 (медиана 2 = среднее — показано), ctrl_f4-версии, медианы seed-членов для A3."""
    out = {}
    for c, mem in COMBOS.items():
        S = comp.stack(df, mem)
        for r in rules.SIMPLE:
            out["ctl_%s_%s" % (c, r)] = (rules.sigma2(r, S, ctx), "+".join(mem))
        mf = tuple("XF" if m == "XJ" else m for m in mem)
        if "XF" in mf:
            out["ctl_%s_mean_lvl_xf" % c] = (rules.sigma2("mean_lvl", comp.stack(df, mf), ctx), "+".join(mf))
    xs = np.column_stack([comp.col(df, "XJ", s) for s in comp.SEEDS])
    ls = np.column_stack([comp.col(df, "LF", s) for s in comp.SEEDS])
    out["ctl_A3_median_10seeds"] = (np.median(np.column_stack([xs, ls]), axis=1), "XJ s42–46 + LF s42–46")
    out["ctl_A3_mean_of_seed_medians"] = ((np.median(xs, axis=1) + np.median(ls, axis=1)) / 2, "медиана 5 seed XJ, медиана 5 seed LF")
    h3 = mix.h3("roll", ctx, comp.col(df, "XF"))
    out["ref_c2_h3_roll_f4"] = (h3, "HAR + ctrl_f4, roll252 (ссылка Ж-204 C, пересчитано из рядов)")
    out["ref_c2_h3_w05"] = (mix.h3("w05", ctx, comp.col(df, "XF")), "HAR + ctrl_f4, равные (ссылка Ж-204 C)")
    return out


def error_table(df) -> pd.DataFrame:
    """Корреляция ошибок X и L (ln-ошибки и потери QLIKE) и вклад её снижения: выигрыш среднего (log) против случая полной корреляции ошибок."""
    rv, out = df["rv"].to_numpy(float), []
    for a, b in (("XJ", "LF"), ("XF", "LF"), ("XJ", "XF")):
        ea, eb = np.log(rv / comp.col(df, a)), np.log(rv / comp.col(df, b))
        qa, qb = crit_e.qlike_series(rv, comp.col(df, a)), crit_e.qlike_series(rv, comp.col(df, b))
        v_mean, v_full = ((ea + eb) / 2).var(), (np.sqrt(ea.var()) + np.sqrt(eb.var())) ** 2 / 4
        qm = crit_e.qlike_series(rv, np.exp((np.log(comp.col(df, a)) + np.log(comp.col(df, b))) / 2))
        out.append({"pair": "%s|%s" % (a, b), "corr_ln_err": float(np.corrcoef(ea, eb)[0, 1]), "corr_qlike": float(np.corrcoef(qa, qb)[0, 1]), "var_mean_err": float(v_mean),
                    "var_if_corr_1": float(v_full), "var_gain_decorrelation": float(v_full - v_mean), "qlike_mean_log": float(qm.mean()),
                    "qlike_best_comp": float(min(qa.mean(), qb.mean())), "qlike_avg_comp": float((qa.mean() + qb.mean()) / 2)})
    return pd.DataFrame(out)


def run_rule(df, ctx, combo: str, rule: str):
    S = comp.stack(df, COMBOS[combo])
    t0 = time.time()
    w = rules.weights(rule, S, ctx)
    return np.einsum("nk,nk->n", w, S), w, time.time() - t0


def stage(_a=None) -> int:
    df = comp.load()
    ctx, base, refs = comp.ctx_of(df), df[["d", "target_date", "rv"]], refs_of(df)
    ser_out, rows, wts, reg = {}, [], {c: [] for c in COMBOS}, []

    def add(name, s2, kind, trial, members, combo="", secs=0.0, w=None, rule=""):
        r = {**crit_e.describe(name, s2, base, refs), "kind": kind, "trial": trial, "members": members, "combo": combo, "rule": rule, "secs": secs}
        comps = [crit_e.describe(m, comp.col(df, m), base)["crit_wf"] for m in COMBOS[combo]] if combo else [np.nan]
        r["best_component_crit_wf"] = float(np.nanmin(comps))
        r["added_value_vs_best_component"] = r["best_component_crit_wf"] - r["crit_wf"]
        rows.append(r)
        ser_out[name] = np.asarray(s2, float)
        if w is not None:
            x = pd.DataFrame(w, columns=["w_" + m for m in COMBOS[combo]])
            x.insert(0, "id", name)
            x.insert(0, "d", df["d"].to_numpy())
            wts[combo].append(x)
        if trial:
            reg.append(pe.trial({"wave": "EA", "combo": combo, "members": members, "rule": rule, "seed": 42}, "EA", "ensemble",
                                "crit_wf=%.6f;qlike_all=%.6f" % (r["crit_wf"], r["qlike_all"]), secs))
    for k in ("XJ", "XF", "LF", "B2X", "B2L", "HAR"):
        add(k, comp.col(df, k), "component", False, comp.NAMES[k])
    for k, (s2, mem) in controls(df, ctx).items():
        add(k, s2, "reference" if k.startswith("ref_") else "control", False, mem)
    for c in COMBOS:
        for r in TRIAL_RULES:
            s2, w, secs = run_rule(df, ctx, c, r)
            add("%s_%s" % (c, r), s2, "trial", True, "+".join(COMBOS[c]), c, secs, w, r)
    best = min((r for r in rows if r["kind"] == "trial"), key=lambda r: (r["crit_wf"], r["id"]))
    for r in ("roll_126", "roll_504"):
        s2, w, secs = run_rule(df, ctx, best["combo"], r)
        add("%s_%s" % (best["combo"], r), s2, "trial", True, "+".join(COMBOS[best["combo"]]), best["combo"], secs, w, r)
    out = pd.DataFrame(rows)
    out["status"] = pe.STATUS
    for c, d in DIRS.items():
        sub = out[(out["combo"] == c) | out["kind"].isin(["component"]) | out["id"].str.startswith("ctl_" + c) | ((c == "A1") & out["id"].str.startswith("ref_"))]
        pe.write(sub, d / ("ENSEMBLE_RESULTS.csv" if c == "A3" else "HYBRID_RESULTS.csv"), "тип A (%s): trials + контроли + компоненты; added_value > 0 — лучше лучшего компонента по crit_wf" % c)
        pd.concat(wts[c]).to_csv(d / "WEIGHTS_BY_DATE.csv", index=False, lineterminator="\n") if wts[c] else None
    pe.write(error_table(df), pe.ENS / "ERROR_CORRELATION.csv", "A3: корреляция ошибок X и L и вклад её снижения")
    ser.save("a", df, ser_out)
    n = pe.register(reg)
    print("G5 A: trials EA =", n, "лучшее", best["id"], "crit_wf %.6f; added_value %.6f" % (best["crit_wf"], best["added_value_vs_best_component"]), flush=True)
    return 0
