"""Ж-214, лист 2, стадия `h2` (V6, остаток после «продолжать»): H2, H3, H5, H6, H8, H9 → плато H10, H11 по У9 (GATES_V2). H12 — резерв, не тратится.
Запускается только если CHECKPOINT.decisions.V6.continue; все 8 конфигураций идут в отчёт; V13 — к своим контролям (Л2: A0; Л3: bestQ и A0); B* для V9 — лучшая по crit_wf из прошедших."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s8 import hs_qfit
from models_v2.s9 import hv2_eval as E, hv2_harq as HQ, hv2_harq2 as HR, hv2_hstage as H1S, hv2_ledger as L, hv2_paths as P, hv2_qlev as QL, hv2_world as W

DIR, SUB = "H2", "harq_j"
MODS = (HQ, HR, QL, W, hs_qfit)
SPEC = {"H1": {"q": "rq"},
        "H2": {"q": "rq", "react": ("J",)}, "H3": {"q": "rq", "react": ("w5",)}, "H5": {"q": "rq", "wins": (1, 99)}, "H6": {"q": "rq", "stress": 80},
        "H8": {"q": "rq", "react": ("J",), "qlike": True}}
KIND = {"H2": "harq_ln_dJ", "H3": "harq_ln_dw", "H5": "harq_ln_d_wins", "H6": "harq_ln_d_stress", "H8": "qlike_harq_dJ", "H9": "harq_levels_bpq",
        "H10": "harq_plateau_1", "H11": "harq_plateau_2"}
LEVELS_L3 = ("H8", "H9")      # Л3-кандидаты (QLIKE-подгонка): контроль bestQ и A0


def neighbors(best: str) -> dict:
    """У9: плато лучшей конфигурации остатка∪H1 (прошедшей V13 к A0)."""
    if best == "H5":
        return {"H10": {**SPEC["H5"], "wins": (0.5, 99.5)}, "H11": {**SPEC["H5"], "wins": (2.5, 97.5)}}
    if best == "H6":
        return {"H10": {**SPEC["H6"], "stress": 70}, "H11": {**SPEC["H6"], "stress": 90}}
    base = SPEC[best] if best in ("H1", "H2", "H3", "H8") else SPEC["H1"]
    return {"H10": {**base, "qwin": 504}, "H11": {**base, "qwin": 1008}}


def reg(n: str, spec) -> str:
    return L.register("harq_j", {"name": n, "kind": KIND[n], "spec": spec if isinstance(spec, dict) else "BPQ в уровнях, реакция >= 0 на [min, max] √RQ, QLIKE", "controls": ["A0", "best_Q"]})


def compute(n: str, w: W.World, spec) -> dict:
    if n == "H9":
        s2, B, ab = HR.run_h9(w)
        cols = ["b0", "u", "v", "b_w", "b_m", "b_J", "beta1Q"]
        return {n: s2, **{"%s__%s" % (n, c): B[:, k] for k, c in enumerate(cols)}, n + "__a": ab[:, 0], n + "__b": ab[:, 1]}
    s2, B, names, npair = HR.run(w, spec)
    return {n: s2, n + "__npairs": npair.astype(float), **{"%s__%s" % (n, c): B[:, k] for k, c in enumerate(names)}}


def coefs_table(n: str, ser: dict, w: W.World) -> pd.DataFrame:
    cols = [c for c in ser if c.startswith(n + "__") and not c.endswith(("npairs", "__a", "__b"))]
    yr = pd.DatetimeIndex(w.d).year.to_numpy()
    return pd.DataFrame([{"model": n, "year": int(Y), **{c.split("__")[1]: float(ser[c][np.flatnonzero(yr == Y)[-1]]) for c in cols}} for Y in sorted(set(yr))])


def stage_h2(_a=None) -> int:
    ctx = E.context()
    dec = json.loads(L.CHECK.read_text())["decisions"]
    if not dec.get("V6", {}).get("continue"):
        print("V13 V6: continue = False — остаток не запускается")
        return 0
    w, bq = ctx["w"], H1S.best_q(ctx)
    ctx["refs"]["H1"] = pd.read_parquet(P.OUT / "H_series.parquet")["H1"].to_numpy(float)
    specs, ser, rows, keys = dict(SPEC), {}, [], {}

    def go(n, spec):
        keys[n] = reg(n, spec)
        run = (lambda x: HR.run_h9(x)) if n == "H9" else (lambda x, s=spec: HR.run(x, s))
        ser[n] = E.cached(DIR, n, MODS, lambda: compute(n, w, spec), ctx)
        ctrl = ("A0", "bestQ", "D0_drop_a", "XJ", "LF", "H1")
        rows.append(E.row(ctx, n, ser[n][n], {"kind": KIND[n], "key": keys[n], "trunc_max_diff": E.trunc_real(ctx, run)}, ctrl))

    for n in ("H2", "H3", "H5", "H6", "H8", "H9"):
        go(n, specs.get(n))
    cands = [r for r in rows if r["d_crit_A0"] <= -E.CRIT and r["years_better_A0"] >= E.MIN_YEARS and r["model"] != "H9"]
    best = min(cands, key=lambda r: r["crit_wf"])["model"] if cands else "H1"
    nb = neighbors(best)
    for n, spec in nb.items():
        go(n, spec)
    specs.update(nb)
    h1 = E.row(ctx, "H1", ctx["refs"]["H1"], {"kind": "harq_ln_d"}, ("A0", "bestQ", "D0_drop_a", "XJ", "LF"))
    t = pd.DataFrame([E.row(ctx, "A0", ctx["refs"]["A0"], {"kind": "контроль"}), h1] + rows)
    P.write_csv(t, "HARQ_J_REST.csv", "V6, остаток (У9): H2…H11; Δ — строка минус контроль, минус — строка лучше; плато вокруг %s; потолок: выбор по crit_wf на всей истории" % best, SUB)
    P.write_csv(pd.concat([coefs_table(n, ser[n], w) for n in ser if n != "H9"]), "HARQ_COEFS_REST.csv", "V6: коэффициенты на последнюю дату года; b_dQ — реакция ln RV_d, b_JQ — реакция скачка, b_wQ5 — недельной, b_dQS — добавка при стрессе", SUB)
    P.write_csv(coefs_table("H9", ser["H9"], w), "HARQ_REACTION_LEVELS.csv", "H9 (уровни, √RQ): u = r(min √RQ), v = r(max √RQ) >= 0, beta1Q = (v−u)/(b−a)", SUB)
    res = {}
    for r in rows:
        res[r["model"]] = E.v13([r], ("bestQ", "A0") if (r["model"] in LEVELS_L3 and bq) else ("A0",), 1, 1)
    passed = [n for n, v in res.items() if v["continue"]]
    allp = {**{r["model"]: r["crit_wf"] for r in rows if r["model"] in passed}, **({"H1": h1["crit_wf"]} if dec["V6"]["L2_H1_H4_vs_A0"]["continue"] else {})}
    E.decide("V6_rest", {"plateau_parent": best, "neighbors": {k: {x: str(y) for x, y in v.items()} for k, v in nb.items()}, "V13_each": {n: v["controls"] for n, v in res.items()},
                         "passed_V13": passed, "unused_trials": L.LIMITS["harq_j"] - len(rows) - 3, "B_star_V9": min(allp, key=allp.get) if allp else None, "crit_passed": allp})
    cols = ["model", "qlike_all", "crit_wf", "d_crit_A0", "d_crit_bestQ", "d_crit_H1", "years_better_A0", "d_qlike_no_240222_A0", "trunc_max_diff"]
    print(t[[c for c in cols if c in t]].round(5).to_string(index=False))
    print("parent:", best, "| passed:", passed, "| B*:", min(allp, key=allp.get) if allp else None, flush=True)
    return 0 if all(r["trunc_max_diff"] == 0 for r in rows) else 1
