"""Ж-212 (HS7, G10): стадия `d` — Component Hybrid. D0 (6 trials): полная линейная QLIKE-спецификация и абляция по одному компоненту; компонент информативен, если
crit_wf(drop-k) − crit_wf(full) >= +0,0005. Продолжение по HS7 (уточнение HS11: контроль D0-full входит в набор, поэтому буквальное сравнение «лучшая D0 лучше D0-full» бессмысленно):
Dn_k (<= 3) для трёх самых информативных; Dx_k (<= 3) — только если Dn_k лучше D0-full на Δcrit_wf <= −0,0005. Формы s_k — D_SHAPES.csv."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s5 import pit
from models_v2.s6.xgb import d_comp
from models_v2.s8 import hs_bstage, hs_ctx, hs_dfit as D, hs_eval, hs_leak, hs_ledger as L, hs_paths as P, hs_store

DIR = "D"
ALL = tuple(range(5))
THR = 0.0005


def reg(kind: str, **kw) -> str:
    return L.register("component", {"kind": kind, "phi": list(D.PHI), "objective": "QLIKE-согласованная подгонка по закрытым парам", **kw})


def x_state(mkt_w):
    """X_t для Dx: fn(F, φ_k) -> [φ_k, терциль RV_w (PIT), ln(RV_w/RV_m), терциль рынка (PIT), стресс Ж-210]; всё пересчитывается из переданного F (тест обрезки)."""
    mter = np.nan_to_num(pit.past_bucket(mkt_w, [1 / 3, 2 / 3]), nan=-1)

    def fn(F, ph):
        ter = np.nan_to_num(pit.past_bucket(np.exp(F.X[:, 1]), [1 / 3, 2 / 3]), nan=-1)
        return np.column_stack([ph, ter, F.X[:, 1] - F.X[:, 2], mter, d_comp.stress_state(F).astype(float)])
    return fn


def mkt_weekly(F) -> np.ndarray:
    import pandas as pd_
    from models_v2 import daily
    idly, _ = daily.build_daily(pd_.read_parquet(P.IMOEX), index_days=None)
    return idly["rv_w"].reindex(pd_.DatetimeIndex(F.dates)).to_numpy(float)


def evaluate(ctx: dict, name: str, s2, extra: dict) -> dict:
    df = ctx["df"]
    full = hs_store.get(DIR, "D0_full")
    r = hs_eval.row(df, name, s2, df["HAR"].to_numpy(float), extra)
    r["d_crit_HAR"], r["years_better_HAR"] = r.pop("d_crit"), r.pop("years_better")
    if full is not None:
        c = hs_eval.row(df, name, s2, full)
        r.update({"d_crit_D0full": c["d_crit"], "d_qlike_D0full": c["d_qlike"], "d_qlike_no_240222_D0full": c["d_qlike_no_240222"], "p_dm_D0full": c["p_dm"],
                  "years_better_D0full": c["years_better"], "sign_ok_2of3_D0full": c["sign_ok_2of3"]})
    return r


def stage_d(_a=None) -> int:
    ctx = hs_ctx.build()
    w, F = ctx["w"], ctx["w"]["F"]
    Phi, y, g = D.phi_matrix(F), D.target(F, w["eps"]), w["grid"]
    dts = hs_bstage.leak_rows(ctx)
    rows = []
    for nm, cols in [("D0_full", ALL)] + [("D0_drop_%s" % D.PHI[k], tuple(j for j in ALL if j != k)) for k in ALL]:
        reg("D0", name=nm, cols=list(cols))
        s2 = hs_store.get(DIR, nm)
        if s2 is None:
            s2 = D.lin_run(F, Phi, y, g, cols)
            hs_store.put(DIR, nm, s2)
        rows.append(evaluate(ctx, nm, s2, {"stage": "D0", "comp_dropped": nm.split("_")[-1] if "drop" in nm else "",
                                           "trunc_max_diff": max(hs_leak.trunc_d(ctx, d, cols=cols) for d in dts)}))
    full = next(r for r in rows if r["model"] == "D0_full")
    for r in rows:
        r["info_crit"] = r["crit_wf"] - full["crit_wf"] if r["comp_dropped"] else np.nan
    info = sorted([(r["info_crit"], r["comp_dropped"]) for r in rows if r["comp_dropped"] and r["info_crit"] >= THR], reverse=True)[:3]
    L.decide("component_D0", {"informative": [(c, round(v, 6)) for v, c in info], "rule": "crit_wf(drop-k) − crit_wf(full) >= +0.0005"})
    print("D0: информативные компоненты:", info, flush=True)
    shapes, mk = [], mkt_weekly(F)
    for _, comp_ in info:
        k = D.PHI.index(comp_)
        nm = "Dn_%s" % comp_
        reg("Dn", name=nm, k=comp_, form="пни глубина 1, 100 деревьев, lr 0.05, QLIKE-gamma; offset = линейная часть D0-full без k + c0")
        s2, sh, q = D.nl_run(F, Phi, y, g, Phi[:, [k]], k, 1)
        hs_store.put(DIR, nm, s2)
        yr = pd.Series(pd.DatetimeIndex(ctx["df"]["d"])).dt.year.to_numpy()
        for Y in sorted(set(yr)):
            i = int(np.flatnonzero(yr == Y)[-1])
            shapes += [{"comp": comp_, "year": int(Y), "phi": float(q[j]), "s": float(sh[i, j])} for j in range(len(q))]
        rn = evaluate(ctx, nm, s2, {"stage": "Dn", "comp_dropped": comp_, "trunc_max_diff": max(hs_leak.trunc_d(ctx, d, k=k) for d in dts)})
        rows.append(rn)
        if rn["d_crit_D0full"] <= -THR and sum(r["stage"] == "Dx" for r in rows) < 3:
            xs = x_state(mk)
            Xs = xs(F, Phi[:, k])
            nx = "Dx_%s" % comp_
            reg("Dx", name=nx, k=comp_, X_t="терциль RV_w, ln(RV_w/RV_m), терциль рынка, стресс Ж-210", depth=2)
            s2x, _, _ = D.nl_run(F, Phi, y, g, Xs, k, 2)
            hs_store.put(DIR, nx, s2x)
            rows.append(evaluate(ctx, nx, s2x, {"stage": "Dx", "comp_dropped": comp_, "trunc_max_diff": max(hs_leak.trunc_d(ctx, d, k=k, depth=2, xs_fn=lambda Fx, k=k: xs(Fx, D.phi_matrix(Fx)[:, k])) for d in dts)}))
    nl = [r for r in rows if r["stage"] in ("Dn", "Dx")]
    win = [r["model"] for r in nl if r["d_crit_D0full"] <= -THR and r["years_better_D0full"] >= hs_eval.MIN_YEARS and r["d_qlike_no_240222_D0full"] < 0]
    L.decide("component", {"nonlinear_beats_linear": win, "n_trials": L.used(L.load())["component"], "conclusion": ("нелинейность на: %s" % win) if win else "компонентной нелинейности нет",
                           "unused_trials": L.LIMITS["component"] - L.used(L.load())["component"]})
    t = pd.DataFrame(rows)
    P.write_csv(t, "COMPONENT_D.csv", "D: Δ к HAR-J и к D0-full (минус — лучше); info_crit — вклад компонента (crit_wf(drop) − crit_wf(full)); Dn/Dx — нелинейная форма на информативных компонентах")
    P.write_csv(pd.DataFrame(shapes), "D_SHAPES.csv", "D: форма s_k(φ_k) на конец года (узлы — квантили φ_k до начала сетки; остальные члены D0-full не включены)")
    print(t[["model", "qlike_all", "crit_wf", "d_crit_HAR", "info_crit", "trunc_max_diff"]].round(5).to_string(index=False), flush=True)
    return 0 if (t["trunc_max_diff"] == 0).all() else 1
