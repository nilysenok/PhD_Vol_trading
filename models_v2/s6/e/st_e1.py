"""E1: ансамбль XGB + LGBM без HAR (GATES_E R4, G4). Контроли (средние, медианы) — не trials; 7 правил + 1 trial (лучшее правило на ctrl_f4 + LGBM)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from models_v2.s6.e import comp, crit_e, pe, rules

TRIAL_RULES = ("fixed_2018", "inv_qlike_252", "inv_rmse_252", "dyn_63", "dyn_126", "dyn_252", "stack_nnls")
SERIES = pe.OUT / "series_e1.parquet"


def refs_of(df) -> dict:
    return {"harj": comp.col(df, "HAR"), "b2x": comp.col(df, "B2X"), "b2l": comp.col(df, "B2L"), "xj": comp.col(df, "XJ"), "lf": comp.col(df, "LF")}


def controls(df, ctx) -> dict:
    """{id: σ̂²}: средние по уровню и логарифму (X, L), медиана 2 (= среднее — показано), медиана 10 seed-членов, среднее медиан 5 seed каждой модели; те же средние на ctrl_f4."""
    S, SF = comp.stack(df, ("XJ", "LF")), comp.stack(df, ("XF", "LF"))
    xs = np.column_stack([comp.col(df, "XJ", s) for s in comp.SEEDS])
    ls = np.column_stack([comp.col(df, "LF", s) for s in comp.SEEDS])
    return {"ctl_mean_lvl": rules.sigma2("mean_lvl", S, ctx), "ctl_mean_log": rules.sigma2("mean_log", S, ctx), "ctl_median_2": rules.sigma2("median", S, ctx),
            "ctl_median_10seeds": np.median(np.column_stack([xs, ls]), axis=1), "ctl_mean_of_seed_medians": (np.median(xs, axis=1) + np.median(ls, axis=1)) / 2,
            "ctl_mean_lvl_xf": rules.sigma2("mean_lvl", SF, ctx), "ctl_mean_log_xf": rules.sigma2("mean_log", SF, ctx)}


def error_table(df) -> pd.DataFrame:
    """Корреляция ошибок X и L (ln-ошибки и потери QLIKE) и вклад её снижения: выигрыш среднего (log) против случая полной корреляции ошибок."""
    rv, out = df["rv"].to_numpy(float), []
    for a, b in (("XJ", "LF"), ("XF", "LF"), ("XJ", "XF")):
        ea, eb = np.log(rv / comp.col(df, a)), np.log(rv / comp.col(df, b))
        qa, qb = crit_e.qlike_series(rv, comp.col(df, a)), crit_e.qlike_series(rv, comp.col(df, b))
        va, vb, cov = ea.var(), eb.var(), np.cov(ea, eb)[0, 1]
        v_mean, v_full = ((ea + eb) / 2).var(), (np.sqrt(va) + np.sqrt(vb)) ** 2 / 4
        qm = crit_e.qlike_series(rv, np.exp((np.log(comp.col(df, a)) + np.log(comp.col(df, b))) / 2))
        out.append({"pair": "%s|%s" % (a, b), "corr_ln_err": float(np.corrcoef(ea, eb)[0, 1]), "corr_qlike": float(np.corrcoef(qa, qb)[0, 1]),
                    "var_mean_err": float(v_mean), "var_if_corr_1": float(v_full), "var_gain_decorrelation": float(v_full - v_mean),
                    "qlike_mean_log": float(qm.mean()), "qlike_best_comp": float(min(qa.mean(), qb.mean())), "qlike_avg_comp": float((qa.mean() + qb.mean()) / 2)})
    return pd.DataFrame(out)


def stage(_a=None) -> int:
    df = comp.load()
    ctx, base, refs = comp.ctx_of(df), df[["d", "target_date", "rv"]], refs_of(df)
    S, rows, ser, secs = comp.stack(df, ("XJ", "LF")), [], {}, {}
    for k in ("XJ", "XF", "LF", "B2X", "B2L", "HAR"):
        ser[k] = comp.col(df, k)
        rows.append({**crit_e.describe(k, ser[k], base, refs), "kind": "component", "trial": False, "members": comp.NAMES[k]})
    for k, s2 in controls(df, ctx).items():
        ser[k] = s2
        rows.append({**crit_e.describe(k, s2, base, refs), "kind": "control", "trial": False, "members": "XJ+LF" if "xf" not in k else "XF+LF"})
    for r in TRIAL_RULES:
        t0 = time.time()
        ser["e1_" + r] = rules.sigma2(r, S, ctx)
        rows.append({**crit_e.describe("e1_" + r, ser["e1_" + r], base, refs), "kind": "e1_rule", "trial": True, "members": "XJ+LF", "secs": time.time() - t0})
    best = min((r for r in rows if r["kind"] == "e1_rule"), key=lambda r: (r["crit_wf"], r["id"]))
    rule = best["id"][3:]
    ser["e1_xf_" + rule] = rules.sigma2(rule, comp.stack(df, ("XF", "LF")), ctx)
    rows.append({**crit_e.describe("e1_xf_" + rule, ser["e1_xf_" + rule], base, refs), "kind": "e1_rule_xf", "trial": True, "members": "XF+LF"})
    out = pd.DataFrame(rows)
    out["best_component_crit_wf"] = min(out.loc[out.id.isin(["XJ", "LF"]), "crit_wf"])
    out["added_value_vs_best_component"] = out["best_component_crit_wf"] - out["crit_wf"]
    out["status"] = pe.STATUS
    pe.write(out, pe.ENS / "ENSEMBLE_RESULTS.csv", "E1: компоненты, контроли (не trials), правила; added_value > 0 — лучше лучшего компонента по crit_wf")
    pe.write(error_table(df), pe.ENS / "ERROR_CORRELATION.csv", "E1: корреляция ошибок и вклад её снижения")
    pd.concat([base, pd.DataFrame(ser)], axis=1).to_parquet(SERIES)
    reg = [pe.trial({"wave": "E1", "members": r["members"], "rule": r["id"], "seed": 42}, "E1", "ensemble",
                    "crit_wf=%.6f;qlike_all=%.6f" % (r["crit_wf"], r["qlike_all"]), r.get("secs", 0.0)) for r in rows if r["trial"]]
    n = pe.register(reg)
    print("G4 E1: trials E =", n, "лучшее правило", rule, "crit_wf %.6f (лучший компонент %.6f)" % (best["crit_wf"], out["best_component_crit_wf"].iloc[0]), flush=True)
    return 0
