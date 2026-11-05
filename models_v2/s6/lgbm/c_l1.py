"""C-L1: одно-осевые развёртки вокруг базы C-L0 — 9 осей × 2 точки (≤ 18 trials направления hp). Ось «с эффектом»: min Δcrit_pen ≤ −EFFECT (объявлено в GATES)."""
from __future__ import annotations

import json

import pandas as pd

from models_v2.s6.lgbm import c_hp, c_l0, c_world
from models_v2.s6.lgbm.c_world import DATA_C, CItem

EFFECT = 0.001          # порог эффекта оси по crit_pen (~0,5 % от 0,208); Δ < 0 меньше порога — шум
AXES_FILE = DATA_C / "C_AXES.json"
SIDE = ("lo", "hi")


def items(W: dict, limit: int = 18) -> list:
    bname, spec, hp = c_l0.base_cfg(W)
    out, seen = [], {tuple(h.label() for h in hp.values())}
    for a in c_hp.AXIS_NAMES:
        for s in (0, 1):
            h = c_hp.move(hp, a, s)
            sig = tuple(x.label() for x in h.values())
            if sig in seen:        # точка совпала с базой/другой точкой — не trial
                continue
            seen.add(sig)
            out.append(CItem("C1_%s_%s" % (a, SIDE[s]), spec, h, "hp", "c_l1", "C-L1 ось %s, %s-точка вокруг базы %s" % (a, SIDE[s], bname)))
    if len(out) > limit:
        raise ValueError("C-L1: %d точек > %d" % (len(out), limit))
    return out


def axis_table(its: list, base: str) -> pd.DataFrame:
    cb = c_world.get(base)[1]["crit"]["crit_pen"]
    rec = []
    for a in c_hp.AXIS_NAMES:
        d = {SIDE[s]: c_world.get("C1_%s_%s" % (a, SIDE[s]))[1]["crit"]["crit_pen"] - cb for s in (0, 1) if any(i.name == "C1_%s_%s" % (a, SIDE[s]) for i in its)}
        best = min(d.values()) if d else float("nan")
        rec.append({"axis": a, "d_crit_lo": d.get("lo", float("nan")), "d_crit_hi": d.get("hi", float("nan")), "best_d_crit": best,
                    "effect": bool(d) and best <= -EFFECT, "best_side": min(d, key=d.get) if d else ""})
    return pd.DataFrame(rec)


def stage(W: dict) -> None:
    bname = json.loads(c_l0.BASE_FILE.read_text())["name"]
    its = items(W)
    c_world.run_batch(W, its)
    print("новых trials hp:", c_world.register([i.name for i in its]), flush=True)
    ax = axis_table(its, bname)
    c_world.write(ax, c_world.tab.REP / "C_L1_AXES.csv", "ось -> Δcrit_pen к базе %s; эффект: best Δ ≤ −%g" % (bname, EFFECT))
    df = c_world.rows(W, [bname] + [i.name for i in its], bname)
    c_world.write(df, c_world.tab.REP / "C_L1_RESULTS.csv", "C-L1: %d точек, база %s (контроль C-L0); потолок: выбор и счёт на одних данных" % (len(its), bname))
    AXES_FILE.write_text(json.dumps({"base": bname, "effect": ax.loc[ax.effect, "axis"].tolist(), "table": ax.to_dict("records")}, ensure_ascii=False, indent=1, default=float))
    print(ax.to_string(), "\nCHECKPOINT:", c_world.checkpoint()["used_stream_c"], flush=True)
