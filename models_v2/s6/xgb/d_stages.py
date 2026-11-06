"""Стадии волны D потока XGB: control (G1), d1 (окна и веса, 12 trials), d1sel (годовой выбор окна по прошлому, процедура). Запуск — `runxd.py --stage ...`."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from models_v2.s6 import paths6
from models_v2.s6.xgb import c_calc, c_rows, c_sel, c_space, cfgx, critx, d_calc, d_comp, d_rows, d_space, worldx

BASE_NAME = {"f4": "ctrl_f4", "j": "x2_best_j"}
CRIT_REF = {"f4": 0.20113541384226136, "j": 0.19692134465327366}


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M")


def setup(w: dict) -> dict:
    """-> {'ex': {база: оценка расширяющегося окна D}, 'bases': {база: base_info}, 'ctx', 'st'}; расширяющееся читает кэш D (или считает)."""
    ex = {b: d_calc.evaluate(w, d_space.DCfg(b)) for b in d_space.BASES}
    return {"ex": ex, "bases": {b: d_rows.base_info(BASE_NAME[b], ex[b]) for b in ex}, "ctx": c_rows.context(w, ex["f4"]), "st": d_comp.stress_state(w["F"])}


def c_tid(base: str) -> str:
    if base == "f4":
        return c_space.trial_id("l2", {})
    return json.loads((cfgx.XGB_REP / "X2_DECISION.json").read_text())["best_trial_id"]


def stage_control(w: dict) -> int:
    """G1: расширяющееся окно D = кэш C (побитно f, s² — вся сетка и на CONTROL_DATES; фолды 1e-12; crit_pen = опора C)."""
    res, ok = {}, True
    for b in d_space.BASES:
        ev, tid = d_calc.evaluate(w, d_space.DCfg(b)), c_tid(b)
        rc = pd.read_parquet(c_calc.DATA / ("raw_%s_s42.parquet" % tid))
        fc = pd.read_parquet(c_calc.DATA / ("loyo_%s_s42.parquet" % tid)).sort_values(["Y", "y"]).reset_index(drop=True)
        same = all(ev["raw"][c].to_numpy(float).tobytes() == rc[c].to_numpy(float).tobytes() for c in ("row", "f", "s2"))
        fdiff = float(np.abs(ev["folds"]["q"].to_numpy(float) - fc["q"].to_numpy(float)).max()) if len(fc) == len(ev["folds"]) else float("inf")
        cp = critx.crit(ev["folds"])["crit_pen"]
        res[BASE_NAME[b]] = {"raw_bitwise_all_grid": same, "folds_max_abs_diff": fdiff, "crit_pen": cp, "crit_pen_ref": CRIT_REF[b],
                             "trial_id_c": tid, "secs": ev["secs"]}
        ok = ok and same and fdiff <= 1e-12 and abs(cp - CRIT_REF[b]) < 1e-8
    paths6.dump_json(res, cfgx.XGB_REP / "CONTROL_D.json")
    print("D1 control:", "OK" if ok else "ПРОВАЛ", json.dumps(res, default=float)[:600])
    return 0 if ok else 1


def run_cfg(w: dict, st: dict, cfg: d_space.DCfg, kind: str, direction: str, extra: dict | None = None) -> tuple:
    ev = d_calc.evaluate(w, cfg)
    r = d_rows.row_d(d_space.name_of(cfg), ev["tid"], ev, st["ctx"], st["bases"][cfg.base], kind, st["st"], direction, extra)
    d_rows.save_row(r)
    d_rows.register([d_rows.trial_rec(ev["tid"], d_space.canon(cfg), direction, d_rows.inner_of(r), ev["secs"], stamp())])
    print(d_rows.short(r), flush=True)
    return ev, r


def stage_d1(w: dict) -> int:
    """D1: rolling 504/756/1008/1260, веса 3/5 лет × две базы (12 trials) + доступность пар по годам."""
    st = setup(w)
    for b in d_space.BASES:        # опоры (expanding, ежедневно) — контроль C, не trial
        d_rows.save_row(d_rows.row_d(BASE_NAME[b], st["ex"][b]["tid"], st["ex"][b], st["ctx"], st["bases"][b], "base", st["st"], "temporal"))
    for cfg in d_space.default_cfgs():
        n, hl = d_space.WINDOWS[cfg.win]
        run_cfg(w, st, cfg, "d1_window", "temporal", {"window_pairs": n, "half_life_rows": hl, "freq": "d"})
    av = pd.DataFrame(d_space.availability(w["F"], d_space.spec_for(w), w["grid"]), columns=["year", "window", "share_dates_full_window"])
    d_rows.write_csv(av, cfgx.XGB_REP / "WINDOW_AVAILABILITY_D.csv", "доля дат года, где доступно не меньше N пар (иначе окно = все пары)")
    print("D1: trials D = %d" % d_rows.n_trials())
    return 0


def stage_d1sel(w: dict) -> int:
    """Q3: для каждого года Y окно с минимальным pen_Y по фолдам y < Y (ничья — expanding). Процедура: trial не добавляется."""
    st = setup(w)
    for b in d_space.BASES:
        evs = {k: d_calc.evaluate(w, d_space.DCfg(b, k)) for k in d_space.D1_WINDOWS}
        evs["expanding"] = st["ex"][b]
        choice, comp = c_sel.annual_select(st["ex"][b]["folds"], {k: evs[k]["folds"] for k in d_space.D1_WINDOWS})
        raw = d_comp.annual_raw(w["F"], {k: v["raw"] for k, v in evs.items()}, choice)
        ev = {"fc": worldx.sigma_frame(w, d_space.spec_for(w), raw), "folds": comp, "raw": raw}
        ch = {str(Y): (v or "expanding") for Y, v in choice.items()}
        r = d_rows.row_d(b + "_annual_window", d_space.digest({"annual_window": b, "choice": ch}), ev, st["ctx"], st["bases"][b], "annual_window", st["st"], "temporal",
                         {"choice": json.dumps(ch), "years_not_expanding": sum(1 for v in choice.values() if v), "procedure_not_trial": True})
        d_rows.save_row(r)
        print(d_rows.short(r), ch, flush=True)
    return 0
