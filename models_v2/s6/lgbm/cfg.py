"""Счёт пакета конфигураций (фолды LOYO + ежедневные прогнозы) с кэшем по ключу (trial + код) и строки таблиц относительно baseline."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from models_v2.s6.lgbm import crit, job, tab, world
from models_v2.s6.lgbm.world import Item


def compute(W: dict, it: Item, p) -> None:
    t0 = time.time()
    c = {"crit_pen": float("nan"), "crit_mean": float("nan"), "worst_fold": float("nan"), "by_year": {}}
    if it.folds:
        c = crit.crit(crit.frame(job.run_folds(p, it.name, it.hp_y), it.name))
    res = job.run_days(p, W["F"], it.name, it.hp_y, W["grid"])
    fc, st = job.assemble(W["F"], it.spec, W["eps"], W["grid"], res, it.name)
    fp, sp = world.paths(it.name)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(fp)
    st.update(key=it.key(), trial_id=it.tid(), crit=c, hp_by_year={str(Y): h.label() for Y, h in sorted(it.hp_y.items())}, seed=it.seed,
              kind=it.spec.kind, wt=list(it.spec.wt), direction=it.direction, desc=it.desc, secs_total=time.time() - t0)
    sp.write_text(json.dumps(st, ensure_ascii=False, default=float))


def run_batch(W: dict, items: list, procs: int = 4) -> None:
    todo = [it for it in items if not world.cached(it)]
    if not todo:
        return
    with job.pool(W["F"], {it.name: it.spec for it in todo}, W["eps"], procs) as p:
        for it in todo:
            compute(W, it, p)
            print("готово", it.name, flush=True)


def get(name: str) -> tuple:
    fp, sp = world.paths(name)
    return pd.read_parquet(fp), json.loads(sp.read_text())


def rows(W: dict, names: list, base: str) -> list:
    """Строки таблицы описания для names; baseline — конфигурация base (DM и перспективность — к ней)."""
    fb, sb = get(base)
    out = []
    for n in names:
        fc, st = get(n)
        meta = {"trial_id": st["trial_id"], "direction": st["direction"], "crit_pen": st["crit"]["crit_pen"],
                "crit_pen_base": sb["crit"]["crit_pen"], "crit_mean": st["crit"]["crit_mean"], "worst_fold": st["crit"]["worst_fold"], "desc": st["desc"]}
        out.append(tab.row(n, st["kind"], fc, W["har"], fb, st, meta))
    return out


def register(names: list) -> int:
    """В реестр trials попадают только конфигурации с direction in {hp, obj_feat} (контроли — 'control' — не trials)."""
    r = []
    for n in names:
        st = get(n)[1]
        if st["direction"] in ("hp", "obj_feat"):
            m = "crit_pen=%.6f;crit_mean=%.6f" % (st["crit"]["crit_pen"], st["crit"]["crit_mean"])
            r.append(tab.trial_row(st["trial_id"], st["direction"], "%s|%s|%s" % (n, st["kind"], st["desc"]), m, st["secs_total"]))
    return tab.register(r)


def flat(names: list, key: str = "crit_pen") -> dict:
    return {n: get(n)[1]["crit"][key] for n in names}


def best(names: list) -> str:
    c = flat(names)
    return min(c, key=lambda n: (c[n] if np.isfinite(c[n]) else np.inf))
