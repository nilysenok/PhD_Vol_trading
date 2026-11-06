"""Стадии волны C потока XGB: control (C1), x1 (C6), x3 (C8) и общий расчёт точки. Запуск — `runxc.py --stage ...`."""
from __future__ import annotations

import json
import time

import pandas as pd

from models_v2.s6 import paths6
from models_v2.s6.xgb import c_calc, c_rows, c_space, cfgx

B_DATA = paths6.OUT / "xgboost"          # кэш волны B — только чтение
X3_POINTS = [("learning_rate", -1.0), ("learning_rate", 1.0), ("min_child_weight", -1.0), ("min_child_weight", 1.0)]


def setup(w: dict) -> tuple:
    """-> (ctx, fams, bases): опоры семей l2 (ctrl_f4) и gamma (x3_gamma_f4) — нулевой сдвиг, из кэша либо счётом."""
    bases = {f: c_calc.evaluate(w, f, {}) for f in ("l2", "gamma")}
    return c_rows.context(w, bases["l2"]), {f: c_rows.family_base(b) for f, b in bases.items()}, bases


def bit_equal(a: pd.DataFrame, b: pd.DataFrame, cols) -> bool:
    return len(a) == len(b) and all(a[c].to_numpy(float).tobytes() == b[c].to_numpy(float).tobytes() for c in cols)


def stage_control(w: dict) -> int:
    """C1: нулевой сдвиг нового конвейера = кэш B побитно (прогнозы f, s², фолды q), crit_pen ctrl_f4 = CONTROL_B."""
    res, ok = {}, True
    for fam, bname in (("l2", "ctrl_f4"), ("gamma", "x3_gamma_f4")):
        ev = c_calc.evaluate(w, fam, {})
        raw_n = pd.read_parquet(c_calc.DATA / ("raw_%s_s42.parquet" % ev["tid"]))
        raw_b, loyo_b = pd.read_parquet(B_DATA / ("raw_%s.parquet" % bname)), pd.read_parquet(B_DATA / ("loyo_%s.parquet" % bname))
        same = bit_equal(raw_n, raw_b, ["row", "f", "s2"]) and bit_equal(ev["folds"], loyo_b.sort_values(["Y", "y"]).reset_index(drop=True), ["Y", "y", "q"])
        res[bname] = {"bitwise_raw_and_folds": same, "crit_pen": c_rows.family_base(ev)["crit_pen"], "trial_id_c": ev["tid"], "secs": ev["secs"]}
        ok = ok and same
    ref = json.loads((cfgx.XGB_REP / "CONTROL_B.json").read_text())["crit_f4"]["crit_pen"]
    ok = ok and abs(res["ctrl_f4"]["crit_pen"] - ref) < 1e-12
    res["crit_pen_ctrl_f4_B"] = ref
    paths6.dump_json(res, cfgx.XGB_REP / "CONTROL_C.json")
    if ok:
        ctx, fams, bases = setup(w)
        for fam, name in (("l2", "ctrl_f4"), ("gamma", "x3_gamma_f4")):
            c_rows.save_row(c_rows.row_c(name, fam, {}, bases[fam], ctx, fams[fam], "control"))
    print("C1:", "OK" if ok else "ПРОВАЛ", json.dumps(res, default=float)[:500])
    return 0 if ok else 1


def inner_of(r: dict) -> str:
    keys = ("qlike_all", "qlike_2017_2021", "qlike_2022", "qlike_2023_2025", "d_harj_no_240222", "d_f4", "d_base")
    return "crit_pen=%.6f;crit_mean=%.6f;d_crit_pen=%.6f;%s" % (r["crit_pen"], r["crit_mean"], r["d_crit_pen"], ";".join("%s=%.6f" % (k, r[k]) for k in keys))


def run_point(w: dict, state: tuple, family: str, moves: dict, kind: str, extra: dict | None = None, reg: bool = True) -> dict:
    """Точка (семья, сдвиги): HP не изменился ни в один год — дубль опоры (не trial); иначе счёт, строка, реестр. -> строка."""
    ctx, fams, _ = state
    name, base = c_space.name_of(family, moves), c_calc.hpc_base(w)
    if c_space.years_changed(base, c_space.apply(base, moves)) == 0:
        r = {"id": name, "kind": kind, "family": family, "moves": json.dumps(moves, sort_keys=True), "duplicate": True, "status": c_rows.STATUS, **(extra or {})}
        c_rows.save_row(r)
        print("дубль опоры (зажим во все годы):", name)
        return r
    ev = c_calc.evaluate(w, family, moves)
    r = c_rows.row_c(name, family, moves, ev, ctx, fams[family], kind, extra)
    c_rows.save_row(r)
    if reg:
        c_rows.register([c_rows.trial_rec(ev["tid"], family, moves, inner_of(r), ev["secs"], time.strftime("%Y-%m-%d %H:%M"))])
    print(c_rows.short(r), flush=True)
    return r


def point_extra(w: dict, axis: str, u: float) -> dict:
    return {"axis": axis, "u": u, "clamped_years": c_space.clamped_years(c_calc.hpc_base(w), axis, u), "hp_2023": c_calc.hp_example(w, {axis: u})}


def stage_x1(w: dict) -> int:
    """C-X1: 18 одно-осевых точек вокруг годового выбора И2 на Б-2+F4 (l2)."""
    state = setup(w)
    for ax, u in c_space.x1_points():
        run_point(w, state, "l2", {ax: u}, "x1", point_extra(w, ax, u))
    print("C-X1: trials реестра C = %d" % c_rows.n_hp_trials())
    return 0


def stage_x3(w: dict) -> int:
    """C-X3: Gamma/QLIKE-вариант (reg:gamma ≡ custom QLIKE, B3) × learning_rate, min_child_weight (4 точки)."""
    state = setup(w)
    for ax, u in X3_POINTS:
        run_point(w, state, "gamma", {ax: u}, "x3", point_extra(w, ax, u))
    print("C-X3: trials реестра C = %d" % c_rows.n_hp_trials())
    return 0
