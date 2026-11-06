"""Ж-212 (HS6, HS11, G9): стадия `c` — Residual QLIKE. Минимальный набор Ca, Cc × XGB, LGBM = 4 trials + C4; при выполненном правиле HS11 — Cb, Cd, Ce, Cs ещё 8.
Контроль направления: лучшая строка B своей модели и HAR-J (оба должны быть побиты на Δ crit_wf <= −0,0005 и в >= 6 из 10 лет). Лучшая из Ca–Ce на дату d — причинное правило (HS6)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from models_v2.s8 import hs_bfit, hs_bstage, hs_cfit, hs_ctx, hs_eval, hs_feat, hs_leak, hs_ledger as L, hs_paths as P, hs_rules, hs_store

DIR, GDIR = "C", "Cg"
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
MONO_D = (0, 0, 0, 0, 0, 1, 1)
ARCH = {
    "Ca": "Ж-206:xgb_lgbm_ensemble:fec343a84fe5 (B1q, XGB) / Ж-206:xgb_lgbm_ensemble:6378404826d8 (B2q, LGBM): аддитивная g без взаимодействий (глубина 1) и фиксированные HP вместо годового выбора И2",
    "Cb": "Ж-211:bounded:* (R2 clip, R3 λ — постфактум-ограничение готовой g) и Ca: штраф μ=0,05 на g внутри обучения (grad += μg, hess += μ)",
    "Cc": "Ж-206:xgb_lgbm_ensemble:fec343a84fe5 / 6378404826d8 и Ж-211:residual_xgb:* / residual_lgbm:* (наборы S3/S34/S35/S345): только 7 «состояний HAR», без рыночных групп, закрытые ошибки HAR как признаки корректора",
    "Cd": "Cc + монотонность (неубывание g) по e_HAR среднее 5 и 22 закрытых; монотонные ограничения в реестре не встречаются",
    "Ce": "Cb (штраф μ=0,05, признаки состояния HAR) и Ж-211:residual_*: регуляризация g к нулю в оптимизации на признаках Cc",
    "Cs": "Ж-211:residual_xgb:b64d71a44769 (R3_wf на g B1q): λ_wf по закрытым на g новой архитектуры (причинный выбор из Ca–Ce)",
}
SPEC = {"Ca": ("B2F4", 0.0, ()), "Cb": ("B2F4", 0.05, ()), "Cc": ("S7", 0.0, ()), "Cd": ("S7", 0.0, MONO_D), "Ce": ("S7", 0.05, ())}
MIN_SET = ("Ca", "Cc")


def cname(lib: str, c: str) -> str:
    return "C_%s_%s" % (lib.lower(), c)


def x_builder(ctx: dict, kind: str):
    gate = hs_bfit.xgb_spec(ctx["w"], None).X
    return (lambda F, lb: gate) if kind == "B2F4" else (lambda F, lb: hs_feat.state_c(F, lb)[0])


def run_c(ctx: dict, lib: str, c: str) -> dict:
    kind, mu, mono = SPEC[c]
    name, w, df = cname(lib, c), ctx["w"], ctx["df"]
    L.register("residual_qlike", {"model": lib, "cfg": c, "features": kind, "mu": mu, "mono": list(mono), "hp": hs_cfit.HP, L.ARCH: ARCH[c]})
    bx, t0 = x_builder(ctx, kind), time.time()
    g = hs_store.get(GDIR, name)
    if g is None:
        g = hs_cfit.run(w["F"], hs_bfit.xgb_spec(w, None).X, bx(w["F"], ctx["lb"]), ctx["lb"], w["eps"], w["grid"], lib, mu, mono)
        hs_store.put(GDIR, name, g, positive=False)
        hs_store.put(DIR, name, df["HAR"].to_numpy(float) * np.exp(g))
    return {"name": name, "lib": lib, "cfg": c, "g": g, "secs": time.time() - t0, "bx": bx, "mu": mu, "mono": mono}


def leak(ctx: dict, r: dict) -> float:
    return max(hs_leak.trunc_c(ctx, r["bx"], r["lib"], r["mu"], r["mono"], d) for d in hs_bstage.leak_rows(ctx))


def best_b(lib: str) -> tuple:
    """Лучшая строка B своей модели по crit_wf и строка Bf: (имя, σ̂²), (имя, σ̂²)."""
    t = pd.read_csv(P.REP / "HAR_FEATURES_B.csv", comment="#")
    t = t[t["family"] == lib]
    b = t.sort_values("crit_wf").iloc[0]["model"]
    f = t[t["blocks"] == "Bf"].iloc[0]["model"]
    return (b, hs_store.get("B", b)), (f, hs_store.get("B", f))


def eval_row(ctx: dict, r: dict, extra: dict | None = None) -> dict:
    df, har = ctx["df"], ctx["df"]["HAR"].to_numpy(float)
    s2 = har * np.exp(r["g"])
    (bn, bs), (fn, fs) = best_b(r["lib"])
    row = hs_eval.row(df, r["name"], s2, har, {"family": r["lib"], "cfg": r["cfg"], "secs": round(r["secs"], 1), **(extra or {})})
    for lab, ref in (("bestB", bs), ("Bf", fs)):
        c = hs_eval.row(df, r["name"], s2, ref)
        row.update({"d_crit_" + lab: c["d_crit"], "d_q_" + lab: c["d_qlike"], "p_dm_" + lab: c["p_dm"], "years_better_" + lab: c["years_better"]})
    row.update({"bestB": bn, "d_crit_HAR": row["d_crit"], "years_better_HAR": row["years_better"], "arch_diff": ARCH[r["cfg"]]})
    return row


def ctl_rule(rows: list) -> dict:
    """HS11 для C: лучшая по crit_wf строка набора обязана побить и лучшую B (своей модели), и HAR-J."""
    best = min(rows, key=lambda x: x["crit_wf"])
    ok_b = best["d_crit_bestB"] <= -hs_eval.CRIT_STEP and best["years_better_bestB"] >= hs_eval.MIN_YEARS
    ok_h = best["d_crit_HAR"] <= -hs_eval.CRIT_STEP and best["years_better_HAR"] >= hs_eval.MIN_YEARS
    return {"continue": bool(ok_b and ok_h), "best": best["model"], "vs_bestB": bool(ok_b), "vs_HAR": bool(ok_h), "d_crit_bestB": best["d_crit_bestB"],
            "d_crit_HAR": best["d_crit_HAR"], "rule": "побить лучшую B своей модели и HAR-J: d_crit <= -0.0005 и лет лучше >= 6 из 10"}


def stage_cs(ctx: dict, lib: str, rs: dict) -> dict:
    """Cs (C3): на дату d — g причинно выбранной из Ca–Ce, затем λ_wf ∈ {0; 0,25; 0,5; 0,75; 1} по закрытым (>= 252; до этого λ = 0 = HAR)."""
    df = ctx["df"]
    L.register("residual_qlike", {"model": lib, "cfg": "Cs", "lambdas": list(LAMBDAS), "base_before_252": "lambda=0 (HAR)", L.ARCH: ARCH["Cs"]})
    names = [c for c in ("Ca", "Cb", "Cc", "Cd", "Ce") if (lib, c) in rs]
    G = np.column_stack([rs[(lib, c)]["g"] for c in names])
    har, d, td = df["HAR"].to_numpy(float), df["d"].to_numpy("datetime64[ns]"), df["target_date"].to_numpy("datetime64[ns]")
    S = har[:, None] * np.exp(G)
    ic = hs_rules.causal_pick(hs_rules.qlike_matrix(df["rv"], S), d, td, base=0)
    gp = hs_rules.pick_series(G, ic)
    SL = har[:, None] * np.exp(np.array(LAMBDAS)[None, :] * gp[:, None])
    il = hs_rules.causal_pick(hs_rules.qlike_matrix(df["rv"], SL), d, td, base=0)
    name = cname(lib, "Cs")
    g = np.array(LAMBDAS)[il] * gp
    hs_store.put(GDIR, name, g, positive=False)
    hs_store.put(GDIR, cname(lib, "pick"), gp, positive=False)
    hs_store.put(DIR, name, har * np.exp(g))
    return {"name": name, "lib": lib, "cfg": "Cs", "g": g, "secs": 0.0, "bx": None, "mu": 0.0, "mono": (), "picked": [names[i] for i in np.bincount(ic, minlength=len(names)).argsort()[::-1][:1]]}


def stage_c(_a=None) -> int:
    ctx = hs_ctx.build()
    rs = {(lib, c): run_c(ctx, lib, c) for lib in ("XGB", "LGBM") for c in MIN_SET}
    rows = [eval_row(ctx, r, {"trunc_max_diff": leak(ctx, r)}) for r in rs.values()]
    dec = ctl_rule(rows)
    L.decide("residual_qlike_min", dec)
    print("HS11 C (минимальный набор):", dec, flush=True)
    if dec["continue"]:
        for lib in ("XGB", "LGBM"):
            for c in ("Cb", "Cd", "Ce"):
                rs[(lib, c)] = run_c(ctx, lib, c)
                rows.append(eval_row(ctx, rs[(lib, c)], {"trunc_max_diff": leak(ctx, rs[(lib, c)])}))
        for lib in ("XGB", "LGBM"):
            r = stage_cs(ctx, lib, rs)
            rows.append(eval_row(ctx, r, {"trunc_max_diff": 0.0, "note": "Cs — функция g Ca–Ce (обрезка проверена на g); λ и выбор — по закрытым"}))
        L.decide("residual_qlike", {**dec, "stop": False, "unused_trials": L.LIMITS["residual_qlike"] - (L.used(L.load())["residual_qlike"])})
    else:
        L.decide("residual_qlike", {**dec, "stop": True, "unused_trials": L.LIMITS["residual_qlike"] - len(rows), "reason": "HS11: правило продолжения не выполнено"})
    t = pd.DataFrame(rows)
    P.write_csv(t, "RESIDUAL_QLIKE_C.csv", "C: f=HAR·exp(g), QLIKE-согласованно; Δ к HAR-J / лучшей B / Bf своей модели (минус — лучше); arch_diff — архитектурное отличие от ключей реестра")
    print(t[["model", "qlike_all", "crit_wf", "d_crit_HAR", "d_crit_bestB", "d_crit_Bf", "years_better_bestB", "trunc_max_diff"]].round(5).to_string(index=False), flush=True)
    return 0 if (t["trunc_max_diff"] == 0).all() else 1
