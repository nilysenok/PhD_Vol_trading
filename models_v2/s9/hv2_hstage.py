"""Ж-214, лист 2, стадия `h` (V6, HARQ-J, ступени Л2–Л3): минимальный набор H1 (harq_ln_d), H4 (harq_rel_d), H7 (qlike_harq_d) — 3 trials из 12 → V13 (У2).
Л2-кандидаты H1, H4 — контроль A0; Л3-кандидат H7 — контроль «лучшая Q» (V5, даже если Q не прошла V13) и A0. Обязательный выход — `harq_j/HARQ_REACTION.csv`.
Остаток (H2, H3, H5, H6, H8–H12) запускается отдельной стадией только при «продолжать»."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s8 import hs_qfit
from models_v2.s9 import hv2_eval as E, hv2_harq as HQ, hv2_ledger as L, hv2_paths as P, hv2_world as W

DIR, SUB = "H", "harq_j"
MODS = (HQ, W, hs_qfit)
SPEC = {"H1": ("harq_ln_d", "rq", False, "ln RV_{t+1}=b0+(b_d+b_dQ·q_t)·ln RV_d+b_w ln RV_w+b_m ln RV_m+b_J ln(1+J), q=z(ln√RQ), OLS, σ²=exp(f+s²/2)"),
        "H4": ("harq_rel_d", "rel", False, "H1 с q=z(ln rv_rel), rv_rel=√(2RQ/n)/RV"),
        "H7": ("qlike_harq_d", "rq", True, "H1 с QLIKE-подгонкой в логарифмах (hs_qfit), σ²=exp(f)")}
CTRL = ("A0", "bestQ", "D0_drop_a", "XJ", "LF", "H1")


def register_all() -> dict:
    return {n: L.register("harq_j", {"name": n, "kind": v[0], "q": v[1], "qlike": v[2], "spec": v[3], "controls": ["A0", "best_Q"]}) for n, v in SPEC.items()}


def compute(n: str, w: W.World) -> dict:
    s2, B, npair = HQ.run(w, HQ.q_series(w, SPEC[n][1]), SPEC[n][2])
    return {n: s2, n + "__npairs": npair.astype(float), **{"%s__%s" % (n, c): B[:, k] for k, c in enumerate(HQ.COLS)}}


def best_q(ctx: dict) -> str | None:
    dec = json.loads(L.CHECK.read_text()).get("decisions", {}).get("V5")
    q = pd.read_parquet(P.OUT / "Q_series.parquet") if (P.OUT / "Q_series.parquet").exists() else None
    if dec is None or q is None or dec["best_Q_by_crit_wf"] not in q.columns:
        return None
    ctx["refs"]["bestQ"] = q[dec["best_Q_by_crit_wf"]].to_numpy(float)
    return dec["best_Q_by_crit_wf"]


def stage_h(_a=None) -> int:
    ctx = E.context()
    w, keys = ctx["w"], register_all()
    bq = best_q(ctx)
    ser = {n: E.cached(DIR, n, MODS, lambda n=n: compute(n, w), ctx) for n in SPEC}
    ctx["refs"]["H1"] = ser["H1"]["H1"]
    rows = [E.row(ctx, n, ser[n][n], {"kind": SPEC[n][0], "key": keys[n], "trunc_max_diff": E.trunc_real(ctx, lambda x, n=n: HQ.run(x, HQ.q_series(x, SPEC[n][1]), SPEC[n][2])),
                                      "n_pairs_last": ser[n][n + "__npairs"][-1]}, CTRL) for n in SPEC]
    refs = [E.row(ctx, k, v, {"kind": "контроль"}, CTRL) for k, v in ctx["refs"].items() if k != "H1"]
    t = pd.DataFrame(refs + rows)
    P.write_csv(t, "HARQ_J.csv", "V6, HARQ-J; Δ — QLIKE/crit_wf(строка) − (контроль), минус — строка лучше; bestQ = %s (V5, минимальный набор); потолок: выбор по crit_wf на всей истории" % bq, SUB)
    react = pd.concat([HQ.reaction(n, w, np.column_stack([ser[n]["%s__%s" % (n, c)] for c in HQ.COLS])) for n in SPEC])
    P.write_csv(react, "HARQ_REACTION.csv", "V6: R(q) = b_d + b_dQ·q при q ∈ {−1; 0; +1} на последнюю дату сетки года; знак b_dQ (BPQ ожидают −); среднее b_dQ за год и доля дней с b_dQ<0", SUB)
    d2 = E.v13([r for r in rows if r["model"] in ("H1", "H4")], ("A0",), len(rows), L.LIMITS["harq_j"])
    d3 = E.v13([r for r in rows if r["model"] == "H7"], ("bestQ", "A0") if bq else ("A0",), len(rows), L.LIMITS["harq_j"])
    passed = [x["best"] for x in (d2, d3) if x["continue"]]
    E.decide("V6", {"L2_H1_H4_vs_A0": d2, "L3_H7": d3, "best_Q_control": bq, "minimal_set": list(SPEC), "continue": bool(passed), "passed_V13": passed,
                    "unused_trials": L.LIMITS["harq_j"] - len(rows)})
    cols = ["model", "qlike_all", "crit_wf", "d_crit_A0", "d_crit_bestQ", "d_crit_XJ", "years_better_A0", "years_better_bestQ", "trunc_max_diff"]
    print(t[[c for c in cols if c in t]].round(5).to_string(index=False))
    print(react[["model", "year", "b_d", "b_dQ", "R_q_minus1", "R_q_plus1", "share_b_dQ_neg"]].round(4).to_string(index=False))
    print("V13 L2:", d2["best"], d2["continue"], d2["controls"], "| L3:", d3["best"], d3["continue"], d3["controls"], flush=True)
    return 0 if all(r["trunc_max_diff"] == 0 for r in rows) else 1
