"""Ж-212 (HS9, HS11, G12): стадия `f` — Mixture of Experts, только после B–E. Сначала тест условной информации третьего эксперта (не trial): QLIKE-Вальд HAC для γ в
σ̂² = GEO(HAR, XJ)·exp(γ(ln LF − ln XJ)) и OOS по прошлому; p Холма >= 0,10 или OOS Δ >= 0 — остаётся двухкомпонентная конструкция (F5, F6 не запускаются).
Минимальный набор F1, F2 (геометрические смеси HAR+XJ, HAR+LF). Продолжение (F3, F4 Hedge) — HS11 с контролем S* = XJ (уточнение: «лучший двухэкспертный F1/F2» — сами F1/F2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s5.d import stats_d
from models_v2.s8 import hs_bstage, hs_ctx, hs_enc, hs_eval, hs_ffit as M, hs_ledger as L, hs_paths as P, hs_store

DIR = "F"


def arrays(ctx: dict) -> dict:
    df = ctx["df"]
    return {"rv": df["rv"].to_numpy(float), "har": df["HAR"].to_numpy(float), "xj": df["XJ"].to_numpy(float), "lf": df["LF"].to_numpy(float),
            "d": df["d"].to_numpy("datetime64[ns]"), "td": df["target_date"].to_numpy("datetime64[ns]")}


def build(kind: str, a: dict, extra: int = 0):
    """Ряд σ̂² конфигурации и её веса. F1/F2 — геометрическая смесь двух; F3/F4 — Hedge двух; F5/F6 — три эксперта."""
    rv, d, td = a["rv"], a["d"], a["td"]
    if kind in ("F1", "F2"):
        s2, w = M.geo_mix(rv, a["har"], a["xj"] if kind == "F1" else a["lf"], d, td, extra)
        return s2, {"w_ml": w}
    if kind in ("F3", "F4"):
        s2, W, eta = M.hedge_mix(rv, np.column_stack([a["har"], a["xj"] if kind == "F3" else a["lf"]]), d, td, extra)
        return s2, {"w_har": W[:, 0], "w_ml": W[:, 1], "eta": eta}
    S3 = np.column_stack([a["har"], a["xj"], a["lf"]])
    if kind == "F6":
        s2, W, eta = M.hedge_mix(rv, S3, d, td, extra)
        return s2, {"w_har": W[:, 0], "w_xj": W[:, 1], "w_lf": W[:, 2], "eta": eta}
    simplex = [(i / 10, j / 10, (10 - i - j) / 10) for i in range(11) for j in range(11 - i)]
    simplex.sort(key=lambda w: (np.abs(np.array(w) - 1 / 3).sum(), w))                    # ничья — ближе к равным весам
    C = np.column_stack([np.exp((np.array(w) * np.log(S3)).sum(axis=1)) for w in simplex])
    i = M.hs_rules.causal_pick(M.hs_rules.qlike_matrix(rv, C), d, td, base=0, extra=extra)
    return M.hs_rules.pick_series(C, i), {"w_har": np.array(simplex)[i, 0], "w_xj": np.array(simplex)[i, 1], "w_lf": np.array(simplex)[i, 2]}


def leak_f(ctx: dict, kind: str) -> float:
    """G14: порча экспертов и RV строк > d не меняет σ̂²[:d+1] (побитно)."""
    a, out = arrays(ctx), 0.0
    s0, _ = build(kind, a)
    rng = np.random.default_rng(13)
    for d in hs_bstage.leak_rows(ctx):
        k = int(np.searchsorted(a["d"], ctx["w"]["F"].dates[d].astype("datetime64[ns]")))
        m = np.ones(len(a["rv"]))
        m[k + 1:] = np.exp(rng.normal(0, 1.0, len(m) - k - 1))
        b = dict(a, **{c: a[c] * m for c in ("rv", "har", "xj", "lf")})
        out = max(out, float(np.max(np.abs(build(kind, b)[0][:k + 1] - s0[:k + 1]))))
    return out


def run_kind(ctx: dict, kind: str, a: dict, desc: str) -> tuple:
    L.register("moe", {"kind": kind, "experts": desc, "mix": "geometric", "selection": "causal on closed (>=252)"})
    s2, w = build(kind, a)
    hs_store.put(DIR, kind, s2)
    df = ctx["df"]
    r = hs_eval.row(df, kind, s2, a["xj"], {"experts": desc, "trunc_max_diff": leak_f(ctx, kind)})
    for lab, ref in (("HAR", a["har"]), ("LF", a["lf"])):
        c = hs_eval.row(df, kind, s2, ref)
        r.update({"d_crit_" + lab: c["d_crit"], "d_qlike_" + lab: c["d_qlike"], "p_dm_" + lab: c["p_dm"], "years_better_" + lab: c["years_better"]})
    return r, w


def third_expert(ctx: dict, a: dict, base: np.ndarray) -> dict:
    x = M.third_expert_x(a["xj"], a["lf"])
    wd = hs_enc.wald(a["rv"], base, x)
    s2, _ = hs_enc.oos_expanding(a["rv"], base, x, a["d"], a["td"])
    oos = hs_enc.dm_vs(a["rv"], base, s2)
    ph = float(stats_d.holm_n([wd["p_joint"]], L.n_final())[0])
    info = bool(ph < 0.10 and oos["delta"] < 0)
    return {"gamma": float(wd["gamma"][0]), "t_hac": float(wd["t"][0]), "p_wald": wd["p_joint"], "p_holm_N": ph, "N": L.n_final(), "oos_delta": oos["delta"],
            "oos_p": oos["p"], "gain_in_sample_ceiling": wd["gain_in_sample"], "conditional_info": info, "base": "GEO(HAR,XJ) = F1"}


def stage_f(_a=None) -> int:
    ctx = hs_ctx.build()
    a = arrays(ctx)
    weights, rows = {}, []
    for kind, desc in (("F1", "HAR+XJ"), ("F2", "HAR+LF")):
        r, weights[kind] = run_kind(ctx, kind, a, desc)
        rows.append(r)
    te = third_expert(ctx, a, hs_store.get(DIR, "F1"))
    P.write_csv(pd.DataFrame([te]), "THIRD_EXPERT_TEST.csv", "F: тест условной информации третьего эксперта (не trial): γ при ln LF − ln XJ в σ̂² = F1·exp(γ·x); QLIKE-Вальд HAC и OOS по прошлому; Холм с N")
    dec = hs_eval.hs11(rows)
    L.decide("moe_min", {**dec, "third_expert": te["conditional_info"], "control": "S* = XJ"})
    print("F: третий эксперт информативен:", te["conditional_info"], "| HS11:", dec, flush=True)
    todo = (["F3", "F4"] if dec["continue"] else []) + (["F5", "F6"] if te["conditional_info"] else [])
    for kind in todo:
        r, weights[kind] = run_kind(ctx, kind, a, {"F3": "HAR+XJ", "F4": "HAR+LF"}.get(kind, "HAR+XJ+LF"))
        rows.append(r)
    L.decide("moe", {"stop": not todo, "ran": [r["model"] for r in rows], "two_components": not te["conditional_info"],
                     "unused_trials": L.LIMITS["moe"] - L.used(L.load())["moe"], "reason": "HS11/HS9"})
    t = pd.DataFrame(rows)
    P.write_csv(t, "MOE_F.csv", "F: геометрические смеси/Hedge экспертов; Δ к XJ (S*, основной контроль), HAR-J и LF; минус — лучше")
    wf = pd.DataFrame({"d": ctx["df"]["d"], **{"%s_%s" % (k, c): v for k, ws in weights.items() for c, v in ws.items()}})
    P.write_csv(wf, "F_WEIGHTS.csv", "F: веса экспертов по датам (причинные)")
    print(t[["model", "qlike_all", "crit_wf", "d_crit", "d_crit_HAR", "years_better", "trunc_max_diff"]].round(5).to_string(index=False), flush=True)
    return 0 if (t["trunc_max_diff"] == 0).all() else 1
