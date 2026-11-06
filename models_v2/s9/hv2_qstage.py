"""Ж-214, лист 2, стадия `q` (V5, QLIKE-HAR-J, ступень Л1): минимальный набор Q1, Q2, Q4, Q5 (4 trials из 8) → правило V13 (У1: лучше A0 И D0_drop_a).
Остаток (Q3, Q6, Q7, Q8) запускается отдельной стадией только при «продолжать»; решение и неизрасходованные trials — в CHECKPOINT.json."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s8 import hs_qfit
from models_v2.s9 import hv2_a0, hv2_eval as E, hv2_ledger as L, hv2_paths as P, hv2_qlev as QL, hv2_qlog as QG, hv2_world as W

DIR, SUB = "Q", "harj_qlike"
MODS = (QL, QG, W, hv2_a0, hs_qfit)
BETA = ("b0", "b_d", "b_w", "b_m", "b_J")
SPEC = {"Q1": ("lin_qlike", "σ²=β0+βd·RVd+βw·RVw+βm·RVm+βJ·J в уровнях, β>=0, QLIKE-подгонка (проекция), закрытые пары, ежедневно", True),
        "Q2": ("lin_qlike_nojump", "Q1 при βJ≡0", False),
        "Q4": ("loglin_qlike_decay", "A0-спецификация (4 φ), QLIKE-подгонка с экспоненциальными весами, h∈{250;500;1000;∞} по закрытым (>=252; ничья — больший h)", None),
        "Q5": ("scale_c_pit", "коэффициенты A0, σ²=c_d·exp(f), c_d=mean RV/exp(f) по закрытым (>=252)", None)}


def register_all() -> dict:
    out = {}
    for n, (kind, desc, _) in SPEC.items():
        out[n] = L.register("qlike_harj", {"name": n, "kind": kind, "spec": desc, L.DIFF: E.D0_DROP_KEY, "controls": ["A0", "D0_drop_a"],
                                           **({"overlap_Ж-204": ["Ж-204:59dec5e1ace5", "Ж-204:942df233d1b2"]} if n == "Q4" else {})})
    return out


def runner(n: str):
    if n in ("Q1", "Q2"):
        return lambda w: QL.run(w, with_j=SPEC[n][2])
    if n == "Q4":
        return lambda w: QG.run_decay(w)
    return lambda w: QG.run_scale(w)


def compute(n: str, w: W.World) -> dict:
    o = runner(n)(w)
    if n in ("Q1", "Q2"):
        s2, B = o
        return {n: s2, **{"%s__%s" % (n, c): B[:, k] for k, c in enumerate(BETA if n == "Q1" else BETA[:-1])}}
    if n == "Q4":
        s2, pick, S = o
        return {n: s2, n + "__pick": pick.astype(float), **{"%s__h%s" % (n, "inf" if np.isinf(h) else int(h)): S[h] for h in QG.HL}}
    return {n: o[0]}


def betas_table(ser: dict, w: W.World) -> pd.DataFrame:
    yr, rows = pd.DatetimeIndex(w.d).year.to_numpy(), []
    for n in ("Q1", "Q2"):
        cols = [c for c in ser[n] if "__" in c]
        for Y in sorted(set(yr)):
            m, last = yr == Y, np.flatnonzero(yr == Y)[-1]
            rows.append({"model": n, "year": int(Y), **{c.split("__")[1]: ser[n][c][last] for c in cols}, **{"share_at_zero_" + c.split("__")[1]: float((ser[n][c][m] <= 1e-9).mean()) for c in cols}})
    return pd.DataFrame(rows)


def hl_table(ser: dict, w: W.World) -> pd.DataFrame:
    yr, pick = pd.DatetimeIndex(w.d).year.to_numpy(), ser["Q4"]["Q4__pick"].astype(int)
    return pd.DataFrame([{"year": int(Y), **{"h_%s" % ("inf" if np.isinf(h) else int(h)): float((pick[yr == Y] == k).mean()) for k, h in enumerate(QG.HL)}} for Y in sorted(set(yr))])


def stage_q(_a=None) -> int:
    ctx = E.context()
    w, keys = ctx["w"], register_all()
    ser = {n: E.cached(DIR, n, MODS, lambda n=n: compute(n, w), ctx) for n in SPEC}
    d0 = ctx["refs"].get("D0_drop_a")
    inf_rel = float(np.max(np.abs(ser["Q4"]["Q4__hinf"] / d0 - 1))) if d0 is not None else float("nan")
    rows = [E.row(ctx, n, ser[n][n], {"kind": SPEC[n][0], "key": keys[n], "trunc_max_diff": E.trunc_real(ctx, runner(n))}) for n in SPEC]
    refs = [E.row(ctx, k, v, {"kind": "контроль"}) for k, v in ctx["refs"].items()]
    t = pd.DataFrame(refs + rows)
    t["h_inf_vs_D0_drop_a_max_rel"] = inf_rel
    P.write_csv(t, "QLIKE_HARJ.csv", "V5, QLIKE-HAR-J; Δ — QLIKE/crit_wf(строка) − (контроль), минус — строка лучше; d_*_ctrl: A0, D0_drop_a, XJ, LF; потолок: best выбран по crit_wf на всей истории", SUB)
    P.write_csv(betas_table(ser, w), "Q_BETAS.csv", "V5: β Q1/Q2 на конец года и доля дней на границе β=0", SUB)
    P.write_csv(hl_table(ser, w), "Q4_H_CHOICE.csv", "V5: доли дней по выбранному полураспаду h Q4 (по закрытым прогнозам)", SUB)
    dec = E.v13(rows, ("A0", "D0_drop_a"), len(rows), L.LIMITS["qlike_harj"])
    best_q = min(rows, key=lambda r: r["crit_wf"])["model"]
    E.decide("V5", {**dec, "minimal_set": list(SPEC), "best_Q_by_crit_wf": best_q, "h_inf_equals_D0_drop_a_max_rel": inf_rel,
                    "B_star_for_V9": best_q if dec["continue"] else None})
    cols = ["model", "qlike_all", "crit_wf", "d_crit_A0", "d_crit_D0_drop_a", "d_crit_XJ", "years_better_A0", "years_better_D0_drop_a", "trunc_max_diff"]
    print(t[[c for c in cols if c in t]].round(5).to_string(index=False))
    print("V13:", dec["best"], "continue =", dec["continue"], dec["controls"], "| Q4(h=inf) vs D0_drop_a:", inf_rel, flush=True)
    return 0 if all(r["trunc_max_diff"] == 0 for r in rows) else 1
