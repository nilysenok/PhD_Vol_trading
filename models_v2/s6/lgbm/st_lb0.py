"""LB0: собственный baseline LightGBM — годовой LOYO-выбор по 15 точкам LightGBM-сетки (grid.lb0_points), критерий crit_pen; рядом — наследованный."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s6 import paths6
from models_v2.s6.lgbm import cfg, crit, grid, job, tab, world
from models_v2.s6.lgbm.world import DATA, Item

OWN, INH = "lb0_own", "lb0_inh"


def folds_all(W: dict, pts: tuple) -> dict:
    fp = DATA / "lb0_folds.parquet"
    if fp.exists():
        df = pd.read_parquet(fp)
        return {i: g.drop(columns="pt").reset_index(drop=True) for i, g in df.groupby("pt")}
    sp = job.base_spec(W, "lb0", "ln")
    out = {}
    with job.pool(W["F"], {"lb0": sp}, W["eps"]) as p:
        for i, pt in enumerate(pts):
            out[i] = crit.frame(job.run_folds(p, "lb0", {Y: pt for Y in job.YEARS}), "lb0")
            print("точка", i, pt.label(), "crit_pen %.5f" % crit.crit(out[i])["crit_pen"], flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    pd.concat([f.assign(pt=i) for i, f in out.items()]).to_parquet(fp)
    return out


def tables(frames: dict, pts: tuple, chosen: dict, pens: pd.DataFrame) -> None:
    g = grid.full_grid()
    rec = []
    for i, pt in enumerate(pts):
        c = crit.crit(frames[i])
        rec.append({"point": i, "grid_no": g.index(pt), "label": pt.label(), "crit_pen_fixed": c["crit_pen"], "crit_mean_fixed": c["crit_mean"],
                    "times_chosen": sum(h == pt for h in chosen.values()), **{"pen_%d" % Y: pens.loc[Y, i] for Y in pens.index}})
    tab.write(pd.DataFrame(rec), tab.REP / "LB0_GRID.csv", "LB0: 15 точек из 216 (seed 42), trials HP бюджета LGBM; fixed — одна точка на все годы Y")
    sel = pd.DataFrame([{"Y": Y, "chosen": h.label(), "pen": float(pens.loc[Y].min())} for Y, h in chosen.items()])
    tab.write(sel, tab.REP / "LB0_SELECTION.csv", "LB0: выбор по году Y (цели < 01.01.Y), ничья — grid.order_key")


def stage(W: dict) -> None:
    paths6.mkdirs()
    pts = grid.lb0_points()
    frames = folds_all(W, pts)
    chosen, pens = crit.select(frames, pts)
    tables(frames, pts, chosen, pens)
    sp = job.base_spec(W, "lb0", "ln")
    own = Item(OWN, sp, chosen, "control", "baseline_own", "LB0: собственный годовой LOYO-выбор по 15 точкам LightGBM-сетки; признаки и цель как Б-2")
    inh = Item(INH, sp, {Y: grid.from_i2(W["i2"][Y]) for Y in job.YEARS}, "control", "baseline_inherited",
               "LightGBM-аналог Б-2 с наследованными HP I2 (сетка 132 XGBoost), пересчёт кодом потока")
    cfg.run_batch(W, [inh, own])
    ref = pd.read_parquet(paths6.OUT / "lightgbm" / "fc_lgbm_b2.parquet")
    dif = float(np.max(np.abs(cfg.get(INH)[0]["sigma2"].to_numpy() / ref["sigma2"].to_numpy() - 1)))
    sub = [frames[pts.index(chosen[Y])] for Y in chosen]
    own_c, pts_c = cfg.get(OWN)[1]["crit"], crit.crit(pd.concat([f[f.Y == Y] for f, Y in zip(sub, chosen)]))
    print("LB0: пересчёт inherited vs сохранённый fc_lgbm_b2: max отн. разность %.3e; crit_pen own %.6f (из фолдов сетки %.6f)"
          % (dif, own_c["crit_pen"], pts_c["crit_pen"]))
    rows = cfg.rows(W, [INH, OWN], INH)
    tab.write(pd.DataFrame(rows), tab.REP / "LB0_RESULTS.csv", "LB0 own рядом с inherited; Δ и перспективность — относительно inherited (crit_pen own смещён вниз выбором по 15 точкам — потолок)")
    reg = [tab.trial_row(grid.trial_id("lb0_point", pt), "hp", "LB0 точка %d: %s" % (i, pt.label()),
                         "crit_pen=%.6f;crit_mean=%.6f" % (crit.crit(frames[i])["crit_pen"], crit.crit(frames[i])["crit_mean"]), 0.0) for i, pt in enumerate(pts)]
    print("новых trials HP:", tab.register(reg), "CHECKPOINT:", tab.checkpoint()["used"])
    fc = cfg.get(OWN)[0]
    q = float(metrics.qlike(fc["rv"], fc["sigma2"]).mean())
    json.dump({"hp_by_year": {str(Y): h.label() for Y, h in chosen.items()}, "qlike_all": q, "dif_inh_vs_saved": dif},
              open(DATA / "lb0_summary.json", "w"), ensure_ascii=False, indent=1)
