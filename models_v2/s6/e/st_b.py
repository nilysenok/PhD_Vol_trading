"""Тип B (R6, G6): B1q/B2q (QLIKE-объектив, offset ln HAR), B1m/B2m (log-MSE на остатке, перенос формы Б-1 Ж-200), B1a (аддитивный) + ≤ 3 абляции лучшего B. Всего ≤ 9 trials.
Запуск: --stage b [--only B1q,B2q,...]; каждый ряд кэшируется (ключ = код + вход + seed), повторный запуск продолжает."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s4 import design
from models_v2.s6 import paths6
from models_v2.s6.e import bfit, comp, crit_e, pe, resid, ser
from models_v2.s6.lgbm import c_hp, grid as lgrid, job, world as lworld
from models_v2.s6.xgb import c_space

MAIN = ("B1q", "B2q", "B1m", "B2m", "B1a")
KIND = {"B1q": "xq", "B2q": "lq", "B1a": "xa", "B1m": "h4", "B2m": "h5"}
DIR = {"1": pe.REP / "har_xgboost", "2": pe.REP / "har_lightgbm"}
DROPS = {"noF4": lambda X: X[:, :28], "noIMOEX": lambda X: np.delete(X, range(22, 28), axis=1), "noMarket": lambda X: np.delete(X, range(11, 28), axis=1)}


def hpy_of(W: dict, kind: str, seed: int) -> dict:
    if kind in ("xq", "xa"):
        return {Y: c_space.to_hpc(W["i2"][Y]) for Y in paths6.YEARS}
    return lworld.with_hp(c_hp.lift_y({Y: lgrid.from_i2(W["i2"][Y]) for Y in job.YEARS}), seed=seed)


def run_one(W: dict, name: str, kind: str, X: np.ndarray, seed: int = 42, procs: int = 4, fresh: bool = False) -> pd.DataFrame:
    """Ряд (d, target_date, rv, sigma2, g) с кэшем; h4/h5 — перенос resid.run_h4/h5 на дизайне с X (HP годового выбора И2)."""
    dz = W["dz"]
    W2 = {**W, "dz": design.Dz(X, dz.y, dz.off)}

    def build():
        if kind in ("h4", "h5"):
            fr = (resid.run_h4 if kind == "h4" else resid.run_h5)(W2, seed, procs)
            fr["g"] = np.log(fr["sigma2"].to_numpy(float)) - dz.off[W["grid"]]
            return fr
        return bfit.run(W2, kind, X, hpy_of(W, kind, seed), seed, procs)
    return build() if fresh else bfit.cached("%s_s%d" % (name, seed), bfit.key(kind, X, seed), build)


def diag(df, name, s2, g, lb) -> dict:
    qh, q = ser.qloss(df, "HAR"), crit_e.qlike_series(df["rv"].to_numpy(float), s2)
    st = df["st_g4"].to_numpy(bool)
    return {"id": name, "corr_g_lnHAR": float(np.corrcoef(g, lb)[0, 1]), "share_abs_g_lt_0.05": float((np.abs(g) < 0.05).mean()), "mean_g": float(g.mean()),
            "sd_g": float(g.std()), "dq_calm_vs_har": float((q - qh)[~st].mean()), "dq_stress_vs_har": float((q - qh)[st].mean()), "n_stress": int(st.sum())}


def stage(a=None) -> int:
    W = resid.build_world()
    err = resid.lb_check(W)
    print("G6 lb: max отн. разность exp(lb) и s2_harj = %.3g" % err, flush=True)
    if err > 1e-9:
        return 1
    df, X0, lb = comp.load(), W["dz"].X, W["dz"].off[W["grid"]]
    only = [s for s in (getattr(a, "only", "") or "").split(",") if s]
    res = {}
    for nm in only or MAIN:
        res[nm] = run_one(W, nm, KIND[nm], X0)
        print("G6 %s: QLIKE %.6f за %.0f с" % (nm, crit_e.qlike_series(res[nm]["rv"].to_numpy(float), res[nm]["sigma2"].to_numpy(float)).mean(), res[nm].get("secs", pd.Series([0])).iloc[0]), flush=True)
    if only:
        return 0
    refs = {"harj": comp.col(df, "HAR"), "b2x": comp.col(df, "B2X"), "b2l": comp.col(df, "B2L"), "xj": comp.col(df, "XJ"), "lf": comp.col(df, "LF")}
    rows = [{**crit_e.describe(k, v["sigma2"].to_numpy(float), df, refs), "kind": "trial", "trial": True, "family": k[1]} for k, v in res.items()]
    best = min(rows, key=lambda r: (r["crit_wf"], r["id"]))["id"]
    for ab, fn in DROPS.items():
        nm = "%s_%s" % (best, ab)
        res[nm] = run_one(W, nm, KIND[best], fn(X0))
        rows.append({**crit_e.describe(nm, res[nm]["sigma2"].to_numpy(float), df, refs), "kind": "ablation", "trial": True, "family": best[1]})
    out = pd.DataFrame(rows)
    out["status"] = pe.STATUS
    out["best_component_crit_wf"] = [min(crit_e.describe("h", comp.col(df, "HAR"), df)["crit_wf"], crit_e.describe("c", comp.col(df, "XJ" if r[1] == "1" else "LF"), df)["crit_wf"]) for r in out["id"]]
    out["added_value_vs_best_component"] = out["best_component_crit_wf"] - out["crit_wf"]
    dg = pd.DataFrame([diag(df, k, v["sigma2"].to_numpy(float), v["g"].to_numpy(float), lb) for k, v in res.items()])
    for f, d in DIR.items():
        pe.write(out[out["family"] == f], d / "RESIDUAL_RESULTS.csv", "тип B%s: residual-гибрид HAR-J(б) × exp(g); added_value > 0 — лучше лучшего компонента (HAR / ML) по crit_wf" % f)
        pe.write(dg[dg["id"].str[1] == f], d / "RESIDUAL_DIAGNOSTICS.csv", "тип B%s: g воспроизводит ли HAR (corr g с ln HAR, доля |g|<0,05), вклад в спокойном/стрессе (Г4)" % f)
    ser.save("b", df, {k: v["sigma2"].to_numpy(float) for k, v in res.items()})
    reg = [pe.trial({"wave": "EB", "name": r["id"], "kind": KIND.get(r["id"], KIND[r["id"].split("_")[0]]), "features": r["id"], "seed": 42}, "EB", "residual",
                    "crit_wf=%.6f;qlike_all=%.6f" % (r["crit_wf"], r["qlike_all"])) for r in rows]
    print("G6 B: trials всего", pe.register(reg), "лучший", best, flush=True)
    return 0
