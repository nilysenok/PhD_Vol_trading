"""Тип C (R6, G7): адаптивные гибриды на лучшей базе. C2 вес ML по терцили прошлой RV_w; C3 по знаку изменения RV_w за 5 дней; C4 по режиму стресса Г4; C5 shrink residual-коррекции
c_t ∈ [0,1] по её прошлой точности (252); C6 переключение HAR/ML по накопленной за 63 закрытых разности потерь. C1 = inv_qlike_252 (тип A, не дублируется). ≤ 5 trials + условие уровня 4."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s5 import pit
from models_v2.s5.d import ens
from models_v2.s6.e import comp, crit_e, pe, resid, rules, ser

MAIN_B = ("B1q", "B2q", "B1m", "B2m", "B1a")
THR_L4 = 0.0005


def groups(W: dict) -> tuple:
    """Группы по прошлому: терциль RV_w (past_bucket, 252 предыдущих без текущего) и знак изменения RV_w за 5 дней (RV_w(d) > RV_w(d−5)); −1 — нет данных."""
    F, g = W["F"], W["grid"]
    rvw = np.exp(F.X[:, 1])
    ter = np.nan_to_num(pit.past_bucket(rvw, [1 / 3, 2 / 3]), nan=-1).astype(int)[g]
    ok = np.isfinite(rvw[g]) & np.isfinite(rvw[g - 5])
    return ter, np.where(ok, (rvw[g] > rvw[g - 5]).astype(int), -1)


def build(ctx: dict, h, ml, sb, ter, chg) -> tuple:
    """-> ({C2…C6: σ̂²}, {C2,C3,C4,C6: веса (n, 2)}, c_t). h — HAR, ml — ML-член, sb — гибрид B для C5. Всё — по закрытым прогнозам и признакам на d."""
    rv, S = ctx["rv"], np.column_stack([h, ml])
    m = ens.closed(ctx)
    wts = {"C2": rules.w_group(rv, S, m, ter), "C3": rules.w_group(rv, S, m, chg), "C4": ens.w_regime(rv, S, m, ctx["st_g4"]), "C6": rules.w_switch(rv, S, m)}
    out = {k: np.einsum("nk,nk->n", w, S) for k, w in wts.items()}
    c_t = rules.c_shrink(rv, h, sb, m)
    out["C5"] = np.exp(np.log(h) + c_t * (np.log(sb) - np.log(h)))
    return out, wts, c_t


def best_of(df, names) -> str:
    return min((n for n in names if n in df.columns), key=lambda n: (ser.crit(df, n), n))


def stage(_a=None) -> int:
    W = resid.build_world()
    df = ser.load()
    ctx, base = comp.ctx_of(df), df[["d", "target_date", "rv"]]
    m, rv = ens.closed(ctx), df["rv"].to_numpy(float)
    ml = best_of(df, ("XJ", "LF"))
    a_names = [c for c in df.columns if c[:2] in ("A1", "A2", "A3", "A4") and "_" in c]
    best_a, best_b = best_of(df, a_names), best_of(df, MAIN_B)
    ter, chg = groups(W)
    refs = {"harj": df["HAR"].to_numpy(float), "ml": df[ml].to_numpy(float), "best_a": df[best_a].to_numpy(float), "best_b": df[best_b].to_numpy(float)}
    out_s, wts, c_t = build(ctx, df["HAR"].to_numpy(float), df[ml].to_numpy(float), df[best_b].to_numpy(float), ter, chg)
    mem = {"C2": "HAR+%s, терциль RV_w" % ml, "C3": "HAR+%s, знак ΔRV_w(5)" % ml, "C4": "HAR+%s, стресс Г4" % ml, "C5": "HAR→%s, shrink c_t" % best_b, "C6": "HAR/%s, переключение 63" % ml}
    rows = [{**crit_e.describe(k, v, base, refs), "kind": "trial", "trial": True, "members": mem[k], "base_ml": ml, "base_b": best_b,
             "added_value_vs_best_a": ser.crit(df, best_a) - crit_e.cw(crit_e.qlike_series(rv, v), df["target_date"]),
             "added_value_vs_best_component": min(ser.crit(df, "HAR"), ser.crit(df, ml) if k != "C5" else ser.crit(df, best_b)) - crit_e.cw(crit_e.qlike_series(rv, v), df["target_date"])}
            for k, v in out_s.items()]
    res = pd.DataFrame(rows)
    res["status"] = pe.STATUS
    pe.write(res, pe.ENS / "ADAPTIVE_RESULTS.csv", "тип C: адаптивные гибриды; added_value > 0 — лучше лучшего компонента/лучшего A по crit_wf")
    wd = pd.concat([pd.DataFrame({"d": df["d"], "id": k, "w_HAR": w[:, 0], "w_ML": w[:, 1]}) for k, w in wts.items()] + [pd.DataFrame({"d": df["d"], "id": "C5", "c_t": c_t})])
    wd.to_csv(pe.ENS / "ADAPTIVE_WEIGHTS_BY_DATE.csv", index=False, lineterminator="\n")
    ser.save("c", df, out_s)
    cb, ca = ser.crit(df, best_b), ser.crit(df, best_a)
    l4 = {"best_a": best_a, "crit_wf_best_a": ca, "best_b": best_b, "crit_wf_best_b": cb, "threshold": THR_L4, "b_beats_a_by_more_than_threshold": bool(ca - cb > THR_L4),
          "launched": False, "reason": "условие входа (B лучше A более чем на 0,0005 по crit_wf И ACCEPT R8 (1)–(4)): " + ("часть 1 выполнена, ACCEPT проверяется в e4" if ca - cb > THR_L4 else "не выполнено (часть 1)")}
    (pe.ENS / "LEVEL4_STATUS.json").write_text(json.dumps(l4, ensure_ascii=False, indent=1))
    reg = [pe.trial({"wave": "EC", "name": r["id"], "base_ml": ml, "base_b": best_b, "seed": 42}, "EC", "adaptive", "crit_wf=%.6f;qlike_all=%.6f" % (r["crit_wf"], r["qlike_all"])) for r in rows]
    print("G7 C: trials всего", pe.register(reg), "ML-база", ml, "best A", best_a, "%.6f" % ca, "best B", best_b, "%.6f" % cb, "L4:", l4["reason"], flush=True)
    return 0
