"""Стадия weights (GATES_HF G6, HF7): сглаживание (EMA, полупериод h ∈ {21, 63}) и ограничение скорости (|Δw| ≤ r, r ∈ {0,005; 0,02}) целевых весов roll_126 / roll_252.
H1 = HAR+XJ (roll_126), H2 = HAR+LF (roll_252), H7 = HAR+XJ+LF (roll_252: EMA 63 и r = 0,02). 10 trials `dyn_weights`; A1–A4 Ж-210 переносятся без пересчёта."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s6.e import comp, rules, ser
from models_v2.s7 import hf_cmp as C, hf_data as D, hf_ledger as L, hf_rules as R

PAIRS = {"H1": (("HAR", "XJ"), "roll_126", ((21, 63), (0.005, 0.02))), "H2": (("HAR", "LF"), "roll_252", ((21, 63), (0.005, 0.02))),
         "H7": (("HAR", "XJ", "LF"), "roll_252", ((63,), (0.02,)))}
SERIES = L.OUT / "series_weights.parquet"
PLATEAU = 0.0005


def smooth(kind: str, W: np.ndarray, par: float) -> np.ndarray:
    return R.ema_weights(W, par) if kind == "ema" else R.rate_limit_weights(W, par)


def sigma_of(S, W, kind, par):
    w = smooth(kind, W, par)
    return R.mix(w, S), w


def trunc_diff(S, rule, kind, par, ctx, cuts=(300, 800, 1500)) -> float:
    """Главный тест обрезки: порча rv и S (все члены) в строках j > K не меняет σ̂² и веса строк k <= K."""
    def run(rv, S_):
        c = {**ctx, "rv": rv}
        return sigma_of(S_, rules.weights(rule, S_, c), kind, par)[0]
    base, worst = run(ctx["rv"], S), 0.0
    for K in cuts:
        rv2, S2 = ctx["rv"].copy(), S.copy()
        rv2[K + 1:] *= 41.0
        S2[K + 1:] *= 0.25
        worst = max(worst, float(np.max(np.abs(run(rv2, S2)[:K + 1] - base[:K + 1]))))
    return worst


def stage(_a=None) -> int:
    df = comp.load()
    ctx, sa = comp.ctx_of(df), pd.read_parquet(ser.file("a"))
    out, rows, wts = {}, [], []
    for h, (mem, rule, (hs, rs)) in PAIRS.items():
        S = comp.stack(df, mem)
        W = rules.weights(rule, S, ctx)
        key = {"H1": "A1_roll_126", "H2": "A2_roll_252", "H7": "A4_roll_252"}[h]
        if not np.array_equal(np.einsum("nk,nk->n", W, S), sa[key].to_numpy(float)):
            raise SystemExit("стоп: перенос %s != series_a" % key)
        rows.append(C.row(df, key, sa[key].to_numpy(float), hyb=h, members="+".join(mem), rule=rule, smooth="none", par=0.0, trial=False, note="перенос Ж-210"))
        for kind, pars in (("ema", hs), ("rate", rs)):
            for par in pars:
                name = "%s_%s_%s%g" % (h, rule, kind, par)
                L.register("dyn_weights", {"stage": "weights", "hyb": h, "members": list(mem), "target": rule, "smooth": kind, "par": par})
                s2, w = sigma_of(S, W, kind, par)
                mx = trunc_diff(S, rule, kind, par, ctx)
                if mx != 0.0:
                    raise SystemExit("стоп: обрезка %s %.3g" % (name, mx))
                out[name] = s2
                rows.append(C.row(df, name, s2, hyb=h, members="+".join(mem), rule=rule, smooth=kind, par=par, trial=True, trunc_max_diff=mx))
                x = pd.DataFrame(w, columns=["w_" + m for m in mem])
                x.insert(0, "id", name)
                x.insert(0, "d", df["d"].to_numpy())
                wts.append(x)
    t = pd.DataFrame(rows)
    t["d_neighbor_crit"] = np.nan
    for (h, k), g in t[t["trial"]].groupby(["hyb", "smooth"]):
        if len(g) > 1:
            t.loc[g.index, "d_neighbor_crit"] = abs(g["crit_wf"].diff().abs().max())
    t["plateau"] = t["d_neighbor_crit"] <= PLATEAU
    t["best_of_hyb"] = False
    for h, g in t[t["trial"]].groupby("hyb"):
        t.loc[g["crit_wf"].idxmin(), "best_of_hyb"] = True
    t["status"] = D.STATUS
    D.write(t, "DYNAMIC_WEIGHTS.csv", "HF7: перенос A1/A2/A4 Ж-210 (trial=False) и 10 новых (EMA, лимит скорости); plateau — |Δcrit_wf| соседей ≤ 0,0005")
    D.write(pd.concat(wts, ignore_index=True), "WEIGHTS_HF_BY_DATE.csv", "веса по датам для новых конфигураций; первый член — HAR-J")
    pd.concat([df[["d", "target_date", "rv"]], pd.DataFrame(out)], axis=1).to_parquet(SERIES)
    print(t[["id", "crit_wf", "dcrit_vs_har", "dcrit_vs_star", "qlike_all", "years_won_vs_xj", "d_neighbor_crit", "plateau", "best_of_hyb"]].round(5).to_string(index=False), flush=True)
    print("WEIGHTS: dyn_weights", L.used(L.load()), flush=True)
    return 0
