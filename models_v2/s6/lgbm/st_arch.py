"""Диагностика архитектуры leaf-wise (контроли, не trials): ограниченные деревья, свободные листья, min_data_in_leaf, 4 seed, листья на 12 датах."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s6 import lgb_extra
from models_v2.s6.lgbm import cfg, fitl, job, tab, world
from models_v2.s6.lgbm import st_cfgs as sc
from models_v2.s6.lgbm.st_lb0 import OWN
from models_v2.s6.lgbm.st_run import _reg
from models_v2.s6.lgbm.world import Item, with_hp

SEEDS = (43, 44, 45, 46)


def best_name() -> str:
    reg = _reg()
    names = [OWN] + [n for n, v in reg.items() if v["role"] in ("single", "combo", "f_all", "ablation")]
    return cfg.best(names)


def base_spec(W: dict, name: str) -> job.Spec:
    if name == OWN:
        return job.base_spec(W, "arch_base", "ln")
    v = _reg()[name]
    return sc.feat_spec(W, "arch_base", sc.group_cols(W, v["groups"]), v["drop"])


def variants(hp: dict) -> dict:
    md = {Y: h.min_data for Y, h in hp.items()}
    return {"A_depth3": with_hp(hp, max_depth=3, num_leaves=8), "A_depth4": with_hp(hp, max_depth=4, num_leaves=16),
            "A_nl63": with_hp(hp, num_leaves=63),
            "A_md_lo": {Y: with_hp({Y: h}, min_data=max(5, round(md[Y] / 3)))[Y] for Y, h in hp.items()},
            "A_md_hi": {Y: with_hp({Y: h}, min_data=md[Y] * 3)[Y] for Y, h in hp.items()}}


def leaf_table(W: dict, sp: job.Spec, hp_y: dict, label: str) -> list:
    out = []
    for d in lgb_extra.fixed_dates(W["F"], W["grid"]):
        hp = hp_y[int(W["F"].dates[d + 1].astype("datetime64[Y]").astype(int)) + 1970]
        m = fitl.fit_l(sp.X, sp.y, job.pairs(W["F"], sp, d), hp, "ln", job.weights(W["F"], sp, job.pairs(W["F"], sp, d), d))
        t = m["fin"].trees_to_dataframe()
        c = t.loc[t["split_feature"].isna(), "count"].to_numpy(float)
        out.append({"config": label, "d": str(W["F"].dates[d])[:10], "k": m["k"], "n_leaves": len(c), "leaf_min": c.min(), "leaf_median": float(np.median(c)),
                    "share_lt_2min": float(np.mean(c < 2 * hp.min_data)), "min_data": hp.min_data, "num_leaves": hp.num_leaves})
    return out


def stage(W: dict) -> None:
    bn = best_name()
    sp = base_spec(W, bn)
    hp = cfg.get(bn)[1]
    hp_y = sc.hp_own()
    items = [Item("arch_base", sp, hp_y, "control", "arch", "лучшая конфигурация потока (%s), перезапуск как контроль" % bn)]
    items += [Item(n, sp, h, "control", "arch", n) for n, h in variants(hp_y).items()]
    items += [Item("arch_s%d" % s, sp, with_hp(hp_y, seed=s), "control", "seed", "seed %d" % s, folds=False, seed=s) for s in SEEDS]
    cfg.run_batch(W, items)
    names = [i.name for i in items]
    df = pd.DataFrame(cfg.rows(W, names, "arch_base"))
    tab.write(df, tab.REP / "ARCH_DIAGNOSTICS.csv", "контроли (trials 0); база сравнения — лучшая конфигурация %s (%s); depth3/4 — ограниченные, nl63 — свободнее, md — min_data ÷3/×3" % (bn, hp["desc"][:40]))
    lt = leaf_table(W, sp, hp_y, "base") + leaf_table(W, sp, world.with_hp(hp_y, max_depth=3, num_leaves=8), "depth3") + \
        leaf_table(W, sp, world.with_hp(hp_y, num_leaves=63), "nl63")
    tab.write(pd.DataFrame(lt), tab.REP / "ARCH_LEAVES.csv", "наблюдений в листьях финальной модели на 12 фиксированных датах (первая d каждого года + 22.02.2022)")
    print("лучшая конфигурация потока:", bn)
