"""Seed и соседи для лучших 2 конфигураций C по crit_pen (контроли, не trials) + годовой выбор из кандидатов C по прошлым целям (потолок).
Seed 43–46 — без фолдов (как B); соседи ±1 шаг по осям (до 4 осей на конфигурацию: сначала оси, отличающиеся от базы, затем с лучшим эффектом L1)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s6.lgbm import c_hp, c_l0, c_l1, c_world
from models_v2.s6.lgbm.c_world import CItem
from models_v2.s6.lgbm.world import with_hp

SEEDS = (43, 44, 45, 46)
MAX_AXES = 4


def candidates() -> list:
    """Имена конфигураций C с crit_pen: база + C1_* + C2_*."""
    names = [json.loads(c_l0.BASE_FILE.read_text())["name"]]
    names += sorted(p.stem[3:] for p in c_world.DATA_C.glob("st_C[12]_*.json"))
    return names


def top2(names: list) -> list:
    c = {n: c_world.get(n)[1]["crit"]["crit_pen"] for n in names}
    return sorted(c, key=lambda n: (c[n], n))[:2]


def hp_of_name(W: dict, name: str, base_hp: dict) -> dict:
    """Точное восстановление HP конфигурации C: C1_<ось>_<lo|hi> = move от базы; C2_* — параметры из desc через c_l2.build; иначе база."""
    from models_v2.s6.lgbm import c_l2
    if name.startswith("C1_"):
        axis, side = name[3:].rsplit("_", 1)
        return c_hp.move(base_hp, axis, c_l1.SIDE.index(side))
    if name.startswith("C2_"):
        return c_l2.build(base_hp, json.loads(c_world.get(name)[1]["desc"].split(": ", 1)[1]))
    return base_hp


def items_for(W: dict, name: str, spec, base_hp: dict, base_y: dict, order: list) -> list:
    hp = hp_of_name(W, name, base_hp)
    axes = [a for a in c_hp.changed_axes(hp, base_y)] + [a for a in order if a not in c_hp.changed_axes(hp, base_y)]
    out = [CItem("N_%s_s%d" % (name, s), spec, with_hp(hp, seed=s), "control", "seed", "seed %d для %s" % (s, name), folds=False, seed=s) for s in SEEDS]
    seen = {tuple(h.label() for h in hp.values())}
    for a in axes[:MAX_AXES]:
        for s in (0, 1):
            h = c_hp.move(hp, a, s)
            sig = tuple(x.label() for x in h.values())
            if sig not in seen:
                seen.add(sig)
                out.append(CItem("N_%s_%s_%s" % (name, a, c_l1.SIDE[s]), spec, h, "control", "neighbor", "сосед %s %s от %s" % (a, c_l1.SIDE[s], name)))
    return out


def annual_pick(W: dict, names: list) -> str:
    """Для года Y — кандидат с min pen_Y (по прошлым целям < 01.01.Y); прогноз года — из прогноза кандидата."""
    pens = pd.DataFrame({n: {int(Y): v for Y, v in c_world.get(n)[1]["crit"]["by_year"].items()} for n in names})
    pick = {Y: pens.loc[Y].idxmin() for Y in pens.index}
    base_fc, base_st = c_world.get(names[0])
    yr = pd.to_datetime(base_fc["target_date"]).dt.year.to_numpy()
    fc = base_fc.copy()
    for Y, n in pick.items():
        m = yr == Y
        fc.loc[m, "sigma2"] = c_world.get(n)[0].loc[m, "sigma2"].to_numpy()
    crit_pick = float(np.mean([pens.loc[Y, n] for Y, n in pick.items()]))
    st = dict(base_st, crit={"crit_pen": crit_pick, "crit_mean": float("nan"), "worst_fold": float("nan"), "by_year": {}}, direction="control",
              desc="годовой выбор из %d кандидатов C по pen_Y (прошлые цели); потолок: min по кандидатам" % len(names),
              hp_by_year={str(Y): n for Y, n in pick.items()}, trial_id="annual_pick")
    fp, sp = c_world.paths("C_annual_pick")
    fc.to_parquet(fp)
    sp.write_text(json.dumps(st, ensure_ascii=False, default=float))
    return "C_annual_pick"


def stage(W: dict) -> None:
    bname, spec, base_hp = c_l0.base_cfg(W)
    ax = json.loads(c_l1.AXES_FILE.read_text())["table"]
    order = [r["axis"] for r in sorted(ax, key=lambda r: r["best_d_crit"] if np.isfinite(r["best_d_crit"]) else 9)]
    names = candidates()
    best2 = top2(names)
    its = []
    for n in best2:
        its += items_for(W, n, spec, base_hp, base_hp, order)
    c_world.run_batch(W, its)
    pick = annual_pick(W, names)
    rows = c_world.rows(W, best2 + [i.name for i in its] + [pick], best2[0])
    c_world.write(rows, c_world.tab.REP / "C_SEEDS_NEIGHBORS.csv", "лучшие 2 по crit_pen: %s; seed 43–46 и соседи ±1 шаг — контроли (trials 0); "
                  "C_annual_pick — годовой выбор из кандидатов, потолок; база сравнения %s" % (best2, best2[0]))
    q = {}
    for n in best2:
        v = [c_world.get(x)[0] for x in [n] + ["N_%s_s%d" % (n, s) for s in SEEDS]]
        q[n] = [float(c_world._q(f).mean()) for f in v]
    print("QLIKE по seed 42–46:", {n: [round(x, 5) for x in v] for n, v in q.items()}, "лучшие 2:", best2, flush=True)
