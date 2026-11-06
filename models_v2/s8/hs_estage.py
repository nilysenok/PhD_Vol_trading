"""Ж-212 (HS8, HS11, G11): стадия `e` — Conditional Correction. g — причинно выбранная поправка Ca–Ce модели-победителя C (по crit_wf строки Cs); E1 = строка Cs (ссылка, не trial).
E2 (4 trials): a_t = logistic(θ0 + θ1 z_t) по одному показателю Z1–Z4; E3 (2) и плато ridge ×10 / ×0,1 (2) — только если лучший E2 устойчив (Δcrit_wf к E1 <= −0,0005, знак ΔQLIKE < 0
в >= 2 из 3 периодов и без 24.02.2022). Контроль — самостоятельный XJ. Нет Cs (правило продолжения C не выполнено) — E не запускается («нет g для коррекции»)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s8 import hs_bstage, hs_ctx, hs_dstage, hs_efit as E, hs_eval, hs_feat, hs_ledger as L, hs_paths as P, hs_store

DIR = "E"
RIDGE = 0.01
ZN = ("Z1_eHAR_m22", "Z2_lnRVd_RVm", "Z3_mkt_dev", "Z4_dq_m22")


def indicators(ctx: dict, lib: str) -> np.ndarray:
    """Z1–Z4 на сетке (n×4); Z4 — расхождение HAR−ML своей модели-победителя."""
    w, F, g = ctx["w"], ctx["w"]["F"], ctx["w"]["grid"]
    be, _ = hs_feat.block_e(F, ctx["lb"])
    bg, _ = hs_feat.block_g(F, ctx["lb"], ctx["ml"][lib])
    mk = E.expanding_dev(np.log(hs_dstage.mkt_weekly(F)))
    return np.column_stack([be[:, 2], F.X[:, 0] - F.X[:, 2], mk, bg[:, 2]])[g]


def winner(ctx: dict) -> tuple:
    t = pd.read_csv(P.REP / "RESIDUAL_QLIKE_C.csv", comment="#")
    cs = t[t["cfg"] == "Cs"]
    if cs.empty:
        return None, None
    lib = cs.sort_values("crit_wf").iloc[0]["family"]
    return lib, cs[cs["family"] == lib].iloc[0]["model"]


def act_series(ctx: dict, Z, g, ridge: float) -> tuple:
    df = ctx["df"]
    rv, har = df["rv"].to_numpy(float), df["HAR"].to_numpy(float)
    d, td = df["d"].to_numpy("datetime64[ns]"), df["target_date"].to_numpy("datetime64[ns]")
    a = E.activity(Z, rv, har, g, d, td, ridge)
    return a, har * np.exp(a * g)


def leak_e(ctx: dict, Z, g, ridge: float, a) -> float:
    """G14 для E: порча rv, HAR, g, Z строк > d -> a[:d+1] побитно прежнее (подгонка видит только закрытые)."""
    df = ctx["df"]
    rv, har = df["rv"].to_numpy(float), df["HAR"].to_numpy(float)
    d_, td = df["d"].to_numpy("datetime64[ns]"), df["target_date"].to_numpy("datetime64[ns]")
    out, rng = 0.0, np.random.default_rng(11)
    for d in hs_bstage.leak_rows(ctx):
        k = int(np.searchsorted(d_, ctx["w"]["F"].dates[d].astype("datetime64[ns]")))
        m = np.ones(len(rv))
        m[k + 1:] = np.exp(rng.normal(0, 1.0, len(rv) - k - 1))
        ac = E.activity(Z * m[:, None], rv * m, har * m, g * m, d_, td, ridge)
        out = max(out, float(np.max(np.abs(ac[:k + 1] - a[:k + 1]))))
    return out


def evaluate(ctx, name, s2, e1, extra) -> dict:
    df = ctx["df"]
    r = hs_eval.row(df, name, s2, e1, extra)                        # контроль E1 (Cs): d_crit, d_qlike, years_better — к E1
    for lab, ref in (("HAR", df["HAR"]), ("Sstar", df["XJ"])):
        c = hs_eval.row(df, name, s2, ref.to_numpy(float))
        r.update({"d_crit_" + lab: c["d_crit"], "d_qlike_" + lab: c["d_qlike"], "p_dm_" + lab: c["p_dm"], "years_better_" + lab: c["years_better"]})
    return r


def run_e(ctx, lib, kind, Zsel, names, g, ridge, e1, tag, extra=None) -> dict:
    L.register("conditional", {"kind": kind, "model_g": tag, "z": list(names), "ridge": ridge, "a": "logistic ∈ [0,1]", "fit": "QLIKE по закрытым, >=252, шаг 5"})
    nm = "%s_%s_%s" % (kind, "+".join(n[:2] for n in names), tag + ("_r%g" % ridge if ridge != RIDGE else ""))
    a, s2 = act_series(ctx, Zsel, g, ridge)
    hs_store.put(DIR, nm, s2)
    r = evaluate(ctx, nm, s2, e1, {"kind": kind, "z": "+".join(names), "ridge": ridge, "g_model": tag, "a_mean": float(a.mean()), "a_sd": float(a.std()),
                                   "trunc_max_diff": leak_e(ctx, Zsel, g, ridge, a), **(extra or {})})
    return r


def stable(r: dict) -> bool:
    return bool(r["d_crit"] <= -hs_eval.CRIT_STEP and r["sign_ok_2of3"] and r["d_qlike_no_240222"] < 0)


def stage_e(_a=None) -> int:
    ctx = hs_ctx.build()
    lib, cs_name = winner(ctx)
    if lib is None:
        L.decide("conditional", {"stop": True, "reason": "нет g для коррекции: правило продолжения C (HS11) не выполнено, Cs не строилась", "unused_trials": L.LIMITS["conditional"]})
        P.write_csv(pd.DataFrame([{"status": "не запускалось", "trials_used": 0, "reason": "нет g для коррекции: правило продолжения C (HS11) не выполнено, Cs не строилась; E1 = Cs не существует"}]),
                    "CONDITIONAL_E.csv", "E: направление не запускалось (HS8: минимальный набор E1 + E2×4 только при выполненном HS11 для C); trials Ж-212 по E = 0")
        print("E: не запускается — нет g для коррекции", flush=True)
        return 0
    other = "LGBM" if lib == "XGB" else "XGB"
    g, g2 = hs_store.get("Cg", "C_%s_pick" % lib.lower()), hs_store.get("Cg", "C_%s_pick" % other.lower())
    e1 = hs_store.get("C", cs_name)
    Z = indicators(ctx, lib)
    rows = [run_e(ctx, lib, "E2", Z[:, [i]], [ZN[i]], g, RIDGE, e1, lib) for i in range(4)]
    best = min(rows, key=lambda r: r["crit_wf"])
    ok = stable(best)
    L.decide("conditional_E2", {"best": best["model"], "stable": ok, "d_crit_E1": best["d_crit"], "years_better_E1": best["years_better"], "rule": "HS8: d_crit<=-0.0005, знак в 2 из 3, без 24.02"})
    print("E2 лучший:", best["model"], "устойчив:", ok, flush=True)
    if ok:
        zi = [ZN.index(n) for n in sorted([r["z"] for r in rows], key=lambda z: next(x["crit_wf"] for x in rows if x["z"] == z))[:2]]
        rows += [run_e(ctx, lib, "E3", Z[:, zi], [ZN[i] for i in zi], gg, RIDGE, e1, t) for gg, t in ((g, lib), (g2, other))]
        bi = ZN.index(best["z"])
        rows += [run_e(ctx, lib, "E2plateau", Z[:, [bi]], [ZN[bi]], g, rg, e1, lib) for rg in (RIDGE * 10, RIDGE * 0.1)]
    L.decide("conditional", {"stop": not ok, "stable": ok, "unused_trials": L.LIMITS["conditional"] - L.used(L.load())["conditional"], "g_from": cs_name})
    t = pd.DataFrame(rows)
    P.write_csv(t, "CONDITIONAL_E.csv", "E: σ̂²=HAR·exp(a_t·g), g — поправка модели-победителя C; Δ к E1 (=Cs, ссылка), к HAR-J и к самостоятельному XJ (S*); минус — лучше")
    print(t[["model", "qlike_all", "crit_wf", "d_crit", "d_crit_Sstar", "years_better", "trunc_max_diff"]].round(5).to_string(index=False), flush=True)
    return 0 if (t["trunc_max_diff"] == 0).all() else 1
