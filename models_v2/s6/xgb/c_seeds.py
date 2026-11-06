"""Контроли волны C (не trials, GATES_C_xgb Q6): seed 42–46 для лучших 2 по crit_pen и для опоры ctrl_f4; соседи ±1 шаг лучших 2."""
from __future__ import annotations

import json

import pandas as pd

from models_v2.s5.b1 import evalrows
from models_v2.s6.xgb import c_calc, c_rows, c_space, c_stages, cfgx

SEEDS = (42, 43, 44, 45, 46)
Q_COLS = ("qlike_all", "qlike_2017_2021", "qlike_2022", "qlike_2023_2025", "qlike_no_240222")
CAND_KINDS = ("x1", "x2_best_j", "x3")


def top2(rows: dict) -> list:
    """Лучшие 2 постоянных сдвига по crit_pen (строки X1 / x2_best_j / X3, без дублей и без x2_annual)."""
    c = [r for r in rows.values() if r.get("kind") in CAND_KINDS and not r.get("duplicate")]
    return sorted(c, key=lambda r: (r["crit_pen"], r["id"]))[:2]


def seed_rows(w: dict, ctx: dict, label: str, family: str, moves: dict) -> list:
    """Ряд прогнозов каждого seed (только daily; seed 42 — из кэша строки) -> метрики QLIKE из ряда."""
    out = []
    for s in SEEDS:
        ev = c_calc.evaluate(w, family, moves, seed=s, folds=(s == 42))
        m = evalrows.row(ev["fc"], ctx["ctrl"])
        out.append({"id": label, "family": family, "moves": json.dumps(moves, sort_keys=True), "seed": s, **{k: m[k] for k in Q_COLS}})
        print("seed", label, s, "qlike %.5f" % m["qlike_all"], flush=True)
    return out


def summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("id", sort=False)
    rows = []
    for k, d in g:
        r = {"id": k, "family": d["family"].iloc[0], "moves": d["moves"].iloc[0], "seed": "mean/sd/min/max"}
        for c in Q_COLS:
            r[c] = "%.5f / %.5f / %.5f / %.5f" % (d[c].mean(), d[c].std(ddof=1), d[c].min(), d[c].max())
        rows.append(r)
    return pd.DataFrame(rows)


def stage_seeds(w: dict) -> int:
    state = c_stages.setup(w)
    ctx = state[0]
    rows = c_rows.load_rows()
    best = top2(rows)
    allseeds = []
    for r in best + [rows["ctrl_f4"]]:
        allseeds += seed_rows(w, ctx, r["id"], r["family"], json.loads(r["moves"]))
    df = pd.DataFrame(allseeds)
    c_rows.write_csv(pd.concat([df, summary(df)], ignore_index=True), cfgx.XGB_REP / "HP_SEEDS_C.csv",
                     "seed 42–46 (контроль, не trials) для лучших 2 по crit_pen и опоры ctrl_f4; sd с ddof=1; строки summary — mean/sd/min/max")
    neigh = neighbor_rows(w, state, best, rows)
    keep = ["id", "parent", "family", "moves", "duplicate", "crit_pen", "d_crit_pen", *Q_COLS, "perspective", "unstable", "status"]
    c_rows.write_csv(pd.DataFrame(neigh).reindex(columns=keep), cfgx.XGB_REP / "HP_NEIGHBORS_C.csv", "соседи ±1 шаг лучших 2 (контроль, не trials); u=0 — опора")
    print("C9: лучшие 2:", [r["id"] for r in best], "trials реестра C = %d (не растут)" % c_rows.n_hp_trials())
    return 0


def neighbor_rows(w: dict, state: tuple, best: list, rows: dict) -> list:
    """Соседи ±1 шаг (u=0 — опора, пропускается). Сосед, совпадающий с уже посчитанной строкой (id), берётся из неё, не пересчитывается и не перезаписывается."""
    out = []
    for r in best:
        for m in c_space.neighbors(json.loads(r["moves"])):
            if not m:
                continue
            old = rows.get(c_space.name_of(r["family"], m))
            if old is not None:
                out.append({**old, "parent": r["id"]})
            else:
                out.append(c_stages.run_point(w, state, r["family"], m, "neighbor", {"parent": r["id"]}, reg=False))
    return out
