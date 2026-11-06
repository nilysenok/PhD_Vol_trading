"""Стадия bound (GATES_HF G4): R1 (перенос Ж-210 B1q/B2q), R2 clip(g, −c, c), R3 λ·g, правила R2_wf / R3_wf / R23_wf по закрытым прогнозам.
Нового обучения нет: g — ряды `b_B1q_s42` (XGB), `b_B2q_s42` (LGBM). Trials: bounded 8 + 2, residual_xgb/lgbm по 3 + 1 + 1."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s6.e import comp, pe
from models_v2.s7 import hf_cmp as C, hf_data as D, hf_ledger as L, hf_rules as R

MODELS = {"xgb": ("B1q", "residual_xgb", "RESIDUAL_XGBOOST.csv", "XJ"), "lgbm": ("B2q", "residual_lgbm", "RESIDUAL_LIGHTGBM.csv", "LF")}
LAMS = (0.25, 0.5, 0.75)
SERIES = L.OUT / "series_bound.parquet"


def wf_fn(cfgs):
    return lambda rv, har, g, d, td: R.wf_select(rv, har, g, R.closed_of(rv, d, td), cfgs)[0]


def load_g(df, bname: str):
    b = pd.read_parquet(pe.OUT / ("b_%s_s42.parquet" % bname))
    if not ((b["d"].to_numpy() == df["d"].to_numpy()).all() and (b["rv"].to_numpy() == df["rv"].to_numpy()).all()):
        raise SystemExit("стоп: сетка %s != components" % bname)
    return b["g"].to_numpy(float), b["sigma2"].to_numpy(float)


def one_model(df, mod: str, out: dict, sel: list) -> list:
    bname, resdir, _f, ref = MODELS[mod]
    rv, har, d, td = df["rv"].to_numpy(float), comp.col(df, "HAR"), df["d"].to_numpy("datetime64[ns]"), df["target_date"].to_numpy("datetime64[ns]")
    g, r1 = load_g(df, bname)
    err = float(np.max(np.abs(R.apply(har, g, None, 1.0) / r1 - 1)))
    if err > 1e-8:
        raise SystemExit("стоп: apply(c=None, λ=1) != R1, %.3g" % err)
    m = R.closed_of(rv, d, td)
    rows = [C.row(df, "R1_" + mod, r1, mod=mod, rule="R1", c="inf", lam=1.0, trial=False, note="перенос Ж-210 %s; max отн. разность apply %.2g" % (bname, err))]
    out["R1_" + mod] = r1

    def add(name, s2, rule, c, lam, direction, cfg):
        L.register(direction, {"stage": "bound", "model": mod, **cfg})
        out[name] = np.asarray(s2, float)
        rows.append(C.row(df, name, s2, mod=mod, rule=rule, c=c, lam=lam, trial=True))
    for c in R.C_GRID:
        add("R2_%s_c%.2f" % (mod, c), R.apply(har, g, c, 1.0), "R2", c, 1.0, "bounded", {"kind": "R2", "c": c})
    for lam in LAMS:
        add("R3_%s_l%.2f" % (mod, lam), R.apply(har, g, None, lam), "R3", "inf", lam, resdir, {"kind": "R3", "lam": lam})
    for rule, cfgs, direction in (("R2_wf", R.cfg_grid(lams=(1.0,)), "bounded"), ("R3_wf", R.cfg_grid(cs=(), lams=R.LAM_GRID), resdir),
                                  ("R23_wf", R.cfg_grid(), resdir)):
        s2, idx = R.wf_select(rv, har, g, m, cfgs)
        add("%s_%s" % (rule, mod), s2, rule, "wf", "wf", direction, {"kind": rule, "grid": [[c, l] for c, l in cfgs], "min_closed": R.MIN_CLOSED})
        mx = C.truncation_max_diff(wf_fn(cfgs), rv, har, g, d, td)
        rows[-1]["trunc_max_diff"] = mx
        if mx != 0.0:
            raise SystemExit("стоп: обрезка %s %.3g" % (rule, mx))
        sel.append(pd.DataFrame({"d": df["d"], "model": mod, "rule": rule, "c": [str(cfgs[i][0]) for i in idx], "lam": [cfgs[i][1] for i in idx]}))
    return rows


def plateau(t: pd.DataFrame) -> pd.DataFrame:
    """Устойчивость к соседям: max |Δ crit_wf| между соседними c (R2) и λ (R3) на модель."""
    t = t.copy()
    t["d_neighbor_crit"] = np.nan
    for mod in t["mod"].unique():
        for rule in ("R2", "R3"):
            ix = t.index[(t["mod"] == mod) & (t["rule"] == rule)]
            v = t.loc[ix, "crit_wf"].to_numpy()
            if len(v) > 1:
                t.loc[ix, "d_neighbor_crit"] = [max(abs(v[i] - v[j]) for j in (i - 1, i + 1) if 0 <= j < len(v)) for i in range(len(v))]
    return t


def stage(_a=None) -> int:
    df = comp.load()
    out, sel = {}, []
    L.OUT.mkdir(parents=True, exist_ok=True)
    for mod, (_b, _r, fname, _ref) in MODELS.items():
        t = plateau(pd.DataFrame(one_model(df, mod, out, sel)))
        t["status"] = D.STATUS
        D.write(t, fname, "R1 перенос, R2 clip, R3 λ, правила по закрытым (HF4/HF5); Δ crit < 0 — лучше опоры; trial=True — новая конфигурация Ж-211")
        k = t.sort_values("crit_wf").iloc[0]
        print("BOUND %s: лучшая %s crit_wf %.6f (HAR %.6f, S* %s) qlike_all %.6f" % (mod, k["id"], k["crit_wf"], k["crit_har"], k["star"], k["qlike_all"]), flush=True)
    pd.concat([df[["d", "target_date", "rv"]], pd.DataFrame(out)], axis=1).to_parquet(SERIES)
    D.write(pd.concat(sel, ignore_index=True), "BOUND_SELECTION_BY_DATE.csv", "выбор (c, λ) на d по закрытым прогнозам; c=0.0 и λ=1 — HAR-J")
    print("BOUND: CHECKPOINT", L.load().query("program == 'Ж-211'").groupby("direction").size().to_dict(), flush=True)
    return 0
