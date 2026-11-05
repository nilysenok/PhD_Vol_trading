"""C: устойчивость лучших 3 строк — 5 seeds (42–46) и соседи ±1 шаг (GATES_C §5). Не trials бюджета."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s5.c import engine_c, mix, space, tab_c

SEEDS = (42, 43, 44, 45, 46)
C1_NEED = {"c1_opt": ("opt",), "c1_f4_range": ("f4_range",), "c1_f4_park": ("f4_park",), "c1_f4_f1": ("f4_f1",)}
DZ_OF = {"f4": "f4", "opt": "f4", "top2": "top2", "f4_range": "f4_range", "f4_park": "f4_park", "f4_f1": "f4_f1"}
NB_DESC_AXES = ("depth", "min_child", "lr")
SEED_OK_MIN = 4
PLATEAU_OK = 0.75


def needs(rid: str) -> tuple:
    return C1_NEED.get(rid) or mix.NEEDS[rid]


def hp_of(booster: str, i2: dict, chosen: dict) -> dict:
    return chosen if booster == "opt" else i2


def row_sigma(rid: str, ctx: dict, b: dict) -> np.ndarray:
    return b[C1_NEED[rid][0]] if rid in C1_NEED else mix.build(rid, ctx, b)


def seed_table(p, w, rids, ctx, ctrl, i2, chosen, get_daily) -> pd.DataFrame:
    """Для каждой строки и seed: qlike_all, d к HAR-J(б) без 24.02.2022; seed-ok — по GATES_C §5."""
    bs = sorted({b for r in rids for b in needs(r)})
    sig = {s: {b: get_daily(p, w, DZ_OF[b], hp_of(b, i2, chosen), s)[0]["sigma2"].to_numpy(float) for b in bs} for s in SEEDS}
    rv, h = ctx["rv"], ctx["h"]
    nf = ctx["td"].astype("datetime64[D]") != np.datetime64("2022-02-24")
    q2 = float(tab_c.metrics.qlike(rv, ctrl["s2_b2"].to_numpy(float)).mean())
    rec = []
    for r in rids:
        for s in SEEDS:
            q = tab_c.metrics.qlike(rv, row_sigma(r, ctx, sig[s]))
            rec.append({"row_id": r, "seed": s, "qlike_all": float(q.mean()), "d_harj_no_240222": float((q - tab_c.metrics.qlike(rv, h))[nf].mean())})
    df = pd.DataFrame(rec)
    df["ok"] = (df["qlike_all"] < q2) & (df["d_harj_no_240222"] <= 0)
    agg = df.groupby("row_id").agg(mean=("qlike_all", "mean"), std=("qlike_all", "std"), min=("qlike_all", "min"), max=("qlike_all", "max"),
                                   n_ok=("ok", "sum")).reset_index()
    agg["seed_ok"] = agg["n_ok"] >= SEED_OK_MIN
    return df.merge(agg, on="row_id")


def plateau_table(p, w, boosters, i2, chosen, crit0: dict, crit_b2: float) -> pd.DataFrame:
    """Соседи каждого бустера: внутренний критерий (55 фолдов); crit0[бустер] — критерий самого бустера."""
    rec = []
    for b in boosters:
        for (ax, sg), hp_y in space.neighbors_by_year(hp_of(b, i2, chosen)).items():
            c = engine_c.crit(engine_c.scores(p, DZ_OF[b], hp_y))
            rec.append({"booster": b, "axis": ax, "dir": sg, "crit": c, "d_booster": c - crit0[b], "d_b2": c - crit_b2})
    df = pd.DataFrame(rec)
    df["better_b2"] = df["crit"] < crit_b2
    ok = df.groupby("booster")["better_b2"].mean().rename("share_better_b2").reset_index()
    ok["plateau_ok"] = ok["share_better_b2"] >= PLATEAU_OK
    return df.merge(ok, on="booster")
