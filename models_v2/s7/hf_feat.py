"""Стадия feat (GATES_HF G7, HF6): признаки корректора R4 — наборы S345/S34/S35/S3 (группы F3 полувариации/доходности, F4 квартичность/vov/range, F5 рынок IMOEX), обратная
абляция лучшего (<= 3) и R23_wf на лучшем. Retrain `bfit.run` (HP годового выбора И2, offset ln HAR + c0, seed 42, ежедневно, расширяющееся окно). Наборы выбраны по EX1 Ж-210 — потолок."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from models_v2.s6.e import bfit, comp, st_b, st_ex1
from models_v2.s7 import hf_bound as B, hf_cmp as C, hf_data as D, hf_ledger as L, hf_rules as R

SETS = {"S345": ("F3", "F4", "F5"), "S34": ("F3", "F4"), "S35": ("F3", "F5"), "S3": ("F3",)}
KIND = {"xgb": "xq", "lgbm": "lq"}
SERIES = L.OUT / "series_feat.parquet"


def matrix(w: dict, groups) -> tuple:
    acc = D.feature_groups()
    cols = [c for g in groups for c in acc[g]]
    X = w["feats"][cols].to_numpy(float)
    if len(X) != len(w["F"].rv):
        raise SystemExit("стоп: feats не по строкам F")
    return X, cols


def retrain(w: dict, mod: str, name: str, X: np.ndarray) -> pd.DataFrame:
    p, kind = L.OUT / ("f_%s_%s.parquet" % (mod, name)), KIND[mod]
    kp = p.with_suffix(".key")
    k = bfit.key(kind, X, 42)
    if p.exists() and kp.exists() and kp.read_text() == k:
        return pd.read_parquet(p)
    t0 = time.time()
    fr = bfit.run(w["W"], kind, X, st_b.hpy_of(w["W"], kind, 42), 42, 4)
    fr.to_parquet(p)
    kp.write_text(k)
    print("FEAT retrain %s %s: %.0f с" % (mod, name, time.time() - t0), flush=True)
    return fr


def leak_check(w: dict, mod: str, X: np.ndarray, days=(400, 1200, 2000)) -> float:
    """Порча признаков/целей/offset во всех строках F позже d не меняет ln σ̂² на d (шаг 1 HF6: доступность на d). -> max |Δ|."""
    W, kind = w["W"], KIND[mod]
    F, dz, hpy = W["F"], W["dz"], st_b.hpy_of(W, kind, 42)
    rvn = np.full(len(F.rv), np.nan)
    rvn[:-1] = F.rv[1:]
    y = np.where(np.isfinite(dz.y), np.maximum(rvn, W["eps"]), np.nan)
    worst = 0.0
    for gi in days:
        d = int(W["grid"][gi])
        out = []
        for bad in (False, True):
            X2, y2, lb2 = X.copy(), y.copy(), dz.off.copy()
            if bad:
                X2[d + 1:], y2[d:], lb2[d + 1:] = X2[d + 1:] * 5 + 3, y2[d:] * 50, lb2[d + 1:] + 2.0
            bfit.init(F, X2, lb2, y2, y2, y2, hpy, kind, 42)
            out.append(bfit.day_job(d)[1]["lnmu"])
        worst = max(worst, abs(out[0] - out[1]))
    return worst


def r23(df, mod: str, g, name: str, direction: str) -> tuple:
    rv, har = df["rv"].to_numpy(float), comp.col(df, "HAR")
    d, td = df["d"].to_numpy("datetime64[ns]"), df["target_date"].to_numpy("datetime64[ns]")
    cfgs = R.cfg_grid()
    L.register(direction, {"stage": "feat", "model": mod, "kind": "R23_wf", "on": name, "grid": [[c, l] for c, l in cfgs]})
    s2 = R.wf_select(rv, har, g, R.closed_of(rv, d, td), cfgs)[0]
    return s2, C.truncation_max_diff(B.wf_fn(cfgs), rv, har, g, d, td)


def one_model(w: dict, mod: str, out: dict, r1: dict) -> list:
    df, direction = w["df"], B.MODELS[mod][1]
    rows, res = [], {}

    def run(name, groups, kind):
        X, cols = matrix(w, groups)
        L.register(direction, {"stage": "feat", "model": mod, "kind": kind, "groups": list(groups), "cols": cols})
        fr = retrain(w, mod, name, X)
        res[name] = fr
        out["%s_%s" % (mod, name)] = fr["sigma2"].to_numpy(float)
        lk = leak_check(w, mod, X)
        q, q1 = D.qs(df, fr["sigma2"]), D.qs(df, r1[mod])
        rows.append(C.row(df, "%s_%s" % (mod, name), fr["sigma2"].to_numpy(float), mod=mod, groups="+".join(groups), kind=kind, n_features=len(cols), leak_max_diff=lk,
                          dq_vs_r1=float(q.mean() - q1.mean()), dq_vs_har=float(q.mean() - D.qs(df, df["HAR"]).mean()),
                          **{"dq_har_%d" % y: float((q - D.qs(df, df["HAR"]))[D.year(df) == y].mean()) for y in range(2017, 2027)}))
    for nm, gr in SETS.items():
        run(nm, gr, "set")
    best = min((r for r in rows), key=lambda r: (r["crit_wf"], r["id"]))["id"].split("_", 1)[1]
    for g in (SETS[best] if len(SETS[best]) > 1 else ())[:3]:
        run("abl_%s_no%s" % (best, g), tuple(x for x in SETS[best] if x != g), "ablation")
    top = min(rows, key=lambda r: (r["crit_wf"], r["id"]))
    nm = top["id"].split("_", 1)[1]
    s2, mx = r23(df, mod, res[nm]["g"].to_numpy(float), nm, direction)
    if mx != 0.0:
        raise SystemExit("стоп: обрезка R23_wf %s %.3g" % (nm, mx))
    out["%s_R23wf_on_%s" % (mod, nm)] = s2
    rows.append(C.row(df, "%s_R23wf_on_%s" % (mod, nm), s2, mod=mod, groups=top["groups"], kind="R23_wf", trunc_max_diff=mx, n_features=top["n_features"]))
    return rows


def stage(_a=None) -> int:
    w = D.world()
    L.OUT.mkdir(parents=True, exist_ok=True)
    r1 = {m: pd.read_parquet(B.SERIES)["R1_" + m].to_numpy(float) for m in KIND}
    out, rows = {}, []
    for mod in KIND:
        rows += one_model(w, mod, out, r1)
    t = pd.DataFrame(rows)
    G = {k: D.group_matrix(w["fg"], matrix(w, v)[1]) for k, v in SETS.items()}
    rr = st_ex1.ridge_rows(w["df"], G, None)
    rr = rr[rr["year"] == "all"].set_index("model")["dq_exp_g_minus_har"]
    by_groups = {"+".join(v): k for k, v in SETS.items()}
    t["ridge_loyo_dq_all"] = [rr.get(by_groups.get(g), np.nan) for g in t["groups"]]
    t["status"] = D.STATUS
    D.write(t, "FEATURE_SELECTION.csv", "HF6: шесть шагов на набор: (1) leak_max_diff, (2) ridge_loyo_dq_all, (3) dq_vs_r1/dq_vs_har, (4) qlike_*, (5) dq_har_YYYY, (6) kind=ablation")
    pd.concat([w["df"][["d", "target_date", "rv"]], pd.DataFrame(out)], axis=1).to_parquet(SERIES)
    print(t[["id", "crit_wf", "dcrit_vs_har", "dcrit_vs_star", "qlike_all", "dq_vs_r1", "leak_max_diff", "ridge_loyo_dq_all"]].round(5).to_string(index=False), flush=True)
    print("FEAT: trials", L.used(L.load()), flush=True)
    return 0
