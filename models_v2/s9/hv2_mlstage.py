"""Ж-214, лист 4, стадия `ml` (V9, ступени Л6 XGB и Л7 LGBM; У24–У28 GATES_V2). Минимальный набор N1x-A0, N1l-A0 → V13 (контроли A0 и S* = XJ) → при «продолжать»: N1x-B*, N1l-B* (B* = H2),
N2x, N2l (компоненты T3), абляции групп Q, S, T, M на лучшей библиотеке, Cb-штраф (μ = 0,05) на лучшей; резерв не тратится. XGB и LGBM раздельно, друг в друга не входят."""
from __future__ import annotations

import dataclasses
import json
import sys

import pandas as pd

from models_v2.s8 import hs_cfit, hs_rules
from models_v2.s9 import hv2_a0 as A0M, hv2_dec as DC, hv2_eval as E, hv2_harq as HQ, hv2_harq2 as HR, hv2_ledger as L, hv2_mlout as O, hv2_msfeat as MS, hv2_nfit as NF, hv2_paths as P, hv2_world as W

DIR, LIBS, MU_CB, MIN_SET = "ML", {"x": "XGB", "l": "LGBM"}, 0.05, ("N1x_A0", "N1l_A0")
BASE_TXT = {"A0": "A0 (HAR-J(б), PIT)", "H2": "B* = H2 (V6, прошла V13, решение дирижёра)", "T3": "T3 (V7, схема T, компоненты)"}


def mods() -> tuple:
    return (NF, MS, hs_cfit, hs_rules, A0M, DC, HR, HQ, W, sys.modules[__name__])


def spec(name: str) -> dict:
    """N1x_A0 | N1l_B | N2x | AblQ_x (без группы Q, база A0) | Cb_<конфигурация> (штраф μ)."""
    if name.startswith("Cb_"):
        return {**spec(name[3:]), "mu": MU_CB, "parent": name[3:]}
    if name.startswith("Abl"):
        return {"kind": "N1", "lib": LIBS[name[-1]], "base": "A0", "groups": tuple(g for g in MS.GROUPS if g != name[3]), "mu": 0.0, "parent": "N1%s_A0" % name[-1]}
    base = "T3" if name.startswith("N2") else ("H2" if name.endswith("_B") else "A0")
    return {"kind": name[:2], "lib": LIBS[name[2]], "base": base, "groups": MS.GROUPS, "mu": 0.0}


def diff_key(lib: str, c: str) -> str:
    df = L.load()
    for k, cfg in zip(df["key"], df["config"]):
        if k.startswith("Ж-212:residual_qlike:"):
            j = json.loads(cfg)
            if j.get("model") == lib and j.get("cfg") == c:
                return k
    raise ValueError("нет ключа реестра Ж-212 residual_qlike %s %s" % (lib, c))


def reg(name: str) -> str:
    sp = spec(name)
    cfg = {"name": name, "kind": sp["kind"], "lib": sp["lib"], "base": BASE_TXT[sp["base"]], "groups": list(sp["groups"]), "mu": sp["mu"], "parent": sp.get("parent"), "hp": hs_cfit.HP,
           "lambdas": list(NF.LAMBDAS), "lambda_rule": "λ_wf по закрытым (>= 252), до этого 0 ≡ база", "features": "ядро (4 HAR-признака + ln база) + группы MS (У24)",
           L.DIFF: diff_key(sp["lib"], "Cc"), "отличие_также_от": diff_key(sp["lib"], "Ca"), "controls": ["XJ", "A0", "LF", "Ж-212 Ca/Cc", "A1_roll_126"]}
    return L.register("structural_ml", cfg)


def run_cfg(dw, B: dict, sp: dict, procs: int = NF.PROCS, **kw) -> dict:
    if sp["kind"] == "N2":
        return NF.run_n2(dw, B["T3"], sp["groups"], sp["lib"], sp["mu"], procs, **kw)
    return NF.run_n1(dw, B[sp["base"]], sp["groups"], sp["lib"], sp["mu"], procs, **kw)


def need(sp: dict) -> tuple:
    return ("A0", "T3") if sp["base"] == "T3" else (sp["base"],)


def trunc(dw, sp: dict, procs: int, tail: int = 300) -> float:
    """Главный тест обрезки на реальных данных: σ̂² последних `tail` дат ≤ d из исходного и испорченного (всё после d) мира, базы пересчитаны из того же мира."""
    def f(x):
        x = dataclasses.replace(x, el=x.el[-tail:])
        return run_cfg(x, MS.bases(x, need=need(sp)), sp, procs)["s2"]
    return max(DC.trunc_diff(f, dw, int(dw.el[k])) for k in E.TRUNC_AT)


def compute(dw, B: dict, sp: dict, procs: int, name: str) -> dict:
    o = run_cfg(dw, B, sp, procs)
    out = {name: o["s2"], name + "__full": o["full"], name + "__lam": o["lam"], name + "__fb": o["fb"].astype(float)}
    out.update({"%s__%s" % (name, k): v for k, v in o.items() if k == "g" or k.startswith("g_")})
    return out


def stage_ml(_a=None, procs: int = NF.PROCS) -> int:
    ctx = E.context()
    dw = MS.build_world(ctx["w"], pd.read_parquet(P.M10), pd.read_parquet(P.IMOEX))
    B = MS.bases(dw)
    chk = O.base_check(ctx, dw, B)
    O.add_refs(ctx, dw, B)
    dec = json.loads(L.CHECK.read_text())["decisions"]
    rows, ser, sps = [], {}, {}

    def go(name: str, parent: str | None = None) -> dict:
        sp, key = spec(name), reg(name)
        if parent:
            ctx["refs"]["parent"] = ser[parent][parent]
        ser[name] = E.cached(DIR, name, mods(), lambda: compute(dw, B, sp, procs, name), ctx)
        r = O.row(ctx, dw, B, name, sp, key, ser[name], trunc(dw, sp, procs), parent)
        ctx["refs"].pop("parent", None)
        sps[name] = sp
        rows.append(r)
        return r

    for n in MIN_SET:
        go(n)
    v13 = E.v13(rows, ("A0", "XJ"), len(rows), L.LIMITS["structural_ml"])
    v13["per_library"] = O.per_library(rows)
    plan = {"stop": not v13["continue"], "ran": list(MIN_SET)}
    if v13["continue"]:
        plan.update(run_rest(go, rows, dec))
    E.decide("V9", {**v13, **plan, "unused_trials": L.LIMITS["structural_ml"] - len(rows), "N2_scheme": "T3 (V7 passed T)", "B_star": "H2"})
    O.write_all(ctx, dw, B, rows, ser, sps, chk, v13)
    print(pd.DataFrame(rows)[["config", "lib", "qlike_all", "crit_wf", "d_crit_A0", "d_crit_XJ", "years_better_A0", "years_better_XJ", "d_qlike_no_240222_A0", "d_qlike_no_240222_XJ", "trunc_max_diff"]].round(5).to_string(index=False))
    print(json.dumps({k: v13[k] for k in ("best", "continue", "controls")}, ensure_ascii=False, default=float), flush=True)
    return 0 if all(r["trunc_max_diff"] == 0 for r in rows) else 1


def run_rest(go, rows: list, dec: dict) -> dict:
    """Остаток V9 после «продолжать» (У28): B* (если есть), N2 (если схема T прошла V7), абляции на лучшей библиотеке (A0-база), Cb на лучшей конфигурации."""
    ran = []
    if dec.get("V6_rest", {}).get("B_star_V9") == "H2":
        ran += ["N1x_B", "N1l_B"]
    if "T" in dec.get("V7", {}).get("passed_V13", []):
        ran += ["N2x", "N2l"]
    for n in ran:
        go(n)
    best = min([r for r in rows if r["config"] in MIN_SET], key=lambda r: r["crit_wf"])["config"]
    for g in MS.GROUPS:
        n = "Abl%s_%s" % (g, best[2])
        go(n, best)
        ran.append(n)
    pool = [r for r in rows if r["kind"] in ("N1", "N2") and not r["config"].startswith(("Abl", "Cb"))]
    top = min(pool, key=lambda r: r["crit_wf"])["config"]
    go("Cb_" + top, top)
    return {"ran": [*MIN_SET, *ran, "Cb_" + top], "best_lib_config": best, "cb_parent": top, "reserve_unused": True}
