"""INVENTORY: строки волн C и D. Смеси пересобираются кодом прежних волн (c/mix.build, d/ens.build) из сохранённых прогнозов членов."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from models_v2.s5.c import ctx_c, engine_c, mix
from models_v2.s5.c import stages as cst
from models_v2.s5.d import ctx_d, ens, select_d, stages_d
from models_v2.s6 import inv, inv_meta, origin, paths6

MEMBER = {"H": ("HAR-J(б)", "har"), "X": ("b2_f4", "fx_xgb_ref"), "L": ("d_lgbm_f4", "d_lgb"), "Xr": ("XGB F4 окно 1008", "fx_xgb_ref"),
          "Xh": ("XGB F4 веса 2 года", "fx_xgb_ref"), "f4": ("b2_f4", "fx_xgb_ref"), "opt": ("c1_opt", "fx_xgb_ref"), "top2": ("top2 (F4+F3)", "fx_xgb_ref")}
SINGLE = {"c1_opt": "fx_xgb_ref", "c1_f4_range": "fx_xgb_ref", "c1_f4_park": "fx_xgb_ref", "c1_f4_f1": "fx_xgb_ref",
          "d_lgbm_f4": "d_lgb", "d_xgb_f4_r1008": "fx_xgb_ref", "d_xgb_f4_hl504": "fx_xgb_ref"}


def _keys(C, name, hp_y, seed=42):
    return C["L"]["key"] + ctx_c.code_sha() + name + engine_c.hp_key(hp_y, seed)


def cpath(C, name, hp_y):
    return ctx_c.DATA / ("fc_%s_%s.parquet" % (name, hashlib.sha256(_keys(C, name, hp_y).encode()).hexdigest()[:14]))


def dpath(C, tag, name, hp_y, lib, win, hl):
    s = C["L"]["key"] + ctx_d.code_sha() + name + engine_c.hp_key(hp_y, 42) + "%s|%d|%g" % (lib, win, hl)
    return ctx_d.DATA / ("fc_%s_s42_%s.parquet" % (tag, hashlib.sha256(s.encode()).hexdigest()[:14]))


def member_files(C, rid: str, members: tuple) -> list:
    chosen, f = cst.chosen_hp(), {}
    f["f4"], f["X"] = cpath(C, "f4", C["i2"]), cpath(C, "f4", C["i2"])
    f["opt"], f["top2"] = cpath(C, "f4", chosen), cpath(C, "top2", C["i2"])
    f["L"], f["Xr"], f["Xh"] = (dpath(C, t, n, C["i2"], lib, w, h) for t, n, lib, w, h in
                                (("lgbm_f4", "f4", "lgbm", 0, 0), ("xgb_f4_r1008", "f4", "xgb", 1008, 0), ("xgb_f4_hl504", "f4", "xgb", 0, 504)))
    f["H"] = paths6.OLD_DATA[0] / "forecasts_p1.parquet"
    return [f[m] for m in members if m in f]


def mix_class(members: tuple) -> dict:
    parts, refs = [], []
    for m in members:
        c = origin.classify_chain(origin.CHAINS[MEMBER[m][1]])
        parts.append("%s %s" % (MEMBER[m][0], c["class"]))
        refs.append(c["refs"])
    return {"class": "смесь (%s)" % " + ".join(parts), "refs": " | ".join(dict.fromkeys(refs)), "chain": "models_v2/s5/c/mix.py::build", "library": "—"}


def _emit(C, rid, sigma, tab, members, src, out, trial_id="", ev="нет"):
    ctrl = C["ctrl"]
    rv = ctrl["rv"].to_numpy(float)
    single = SINGLE.get(rid)
    fl = member_files(C, rid, members)
    m = inv_meta.fam(single or "naive", objective="reg:squarederror (L2 на ln-цели)" if single else "смесь над σ̂² членов (правило — код mix/ens)",
                     target=inv_meta.B2T if single else "σ̂² смеси", features="Б-2 + F4 (5 признаков) / top2" if single else "—", hp="I2 по годам (или Optuna TPE 22 для c1_opt)" if single else "—",
                     window="расширяющееся", retrain="ежедневно", es=inv_meta.I1, seed="42")
    if single == "d_lgb":
        m["hp"] = "наследованные HP I2 из сетки 132 XGBoost (lgbfit.params)"
    cls = inv_meta.klass(m) if single else mix_class(members)
    if single and rid == "c1_opt":
        m["hp"] = "TPE 22 trials + I2 по годам (c/tpe.py)"
    if single and rid.startswith("c1_"):
        nm, hp = ("f4", cst.chosen_hp()) if rid == "c1_opt" else (rid[3:], C["i2"])
        fl = [cpath(C, nm, hp)]
    f = [paths6.rel(p) for p in fl]
    r = inv.row(rid, m, cls, "; ".join(f), sigma, rv, family=src, note="" if single else "смесь не сохранена: пересобрана кодом c/mix.build / d/ens.build из прогнозов членов (пути)",
                trial_id=trial_id, eval_selected="да" if rid in select_d.EVAL_SELECTED else ev, qlike_table=tab,
                fsha=";".join(inv.sha(p) for p in fl))
    out.append(r)


HAR_ROWS = ("c3_r4_f4", "c3_r4_opt", "c3_str_f4", "c3_f3calm_har")


def c_members(rid: str) -> tuple:
    return tuple(mix.NEEDS[rid]) + (("H",) if rid.startswith("c2_h3") or rid in HAR_ROWS else ())


def rows() -> list:
    C = ctx_c.load()
    ctx = mix.context(C["w"].F, C["w"].grid, C["ctrl"])
    b = cst.boosters(C, None, ("f4", "opt", "top2"))
    call = paths6.read_csv(paths6.OLD_REP / "C_ALL.csv").set_index("row_id")
    out = []
    for rid in call.index:
        q = float(call.loc[rid, "qlike_all"])
        tid = str(call.loc[rid, "trial_id"])
        if rid == "c2_h2l4_f4_c05":
            out.append(inv.row(rid, inv_meta.fam("naive"), {"class": "смесь (HAR-J(б) + XGBoost H2-L4, c=0,5)", "refs": "", "chain": "models_v2/s5/c/h2l4.py::run", "library": "—"}, "", None, None,
                               family="C_ALL", trial_id=tid, qlike_table=q, note="прогноз не сохранён по имени: h2l4.run() обучает заново — пересчёт не выполнен"))
            continue
        if rid.startswith("c1_"):
            sg = b["opt"] if rid == "c1_opt" else cst.daily(C, None, {"c1_f4_range": "f4_range", "c1_f4_park": "f4_park", "c1_f4_f1": "f4_f1"}[rid], C["i2"])[0]["sigma2"].to_numpy(float)
            _emit(C, rid, sg, q, (), "C_ALL", out, tid)
        else:
            _emit(C, rid, mix.build(rid, ctx, b), q, c_members(rid), "C_ALL", out, tid)
    S = stages_d.Src(C)
    M = {"X": S.sig("f4"), "L": S.sig("L"), "Xr": S.sig("Xr"), "Xh": S.sig("Xh")}
    dres = paths6.read_csv(paths6.OLD_REP / "ENSEMBLE_RESULTS.csv").set_index("row_id")
    for k, rid in zip(("L", "Xr", "Xh"), stages_d.tab_d.COMPONENTS):
        _emit(C, rid, M[k], float(dres.loc[rid, "qlike_all"]), (k,), "ENSEMBLE_RESULTS", out, str(dres.loc[rid, "trial_id"]))
    for rid, (mem, _) in ens.SPEC.items():
        _emit(C, rid, ens.build(rid, ctx, M), float(dres.loc[rid, "qlike_all"]), mem, "ENSEMBLE_RESULTS", out, str(dres.loc[rid, "trial_id"]))
    return out
