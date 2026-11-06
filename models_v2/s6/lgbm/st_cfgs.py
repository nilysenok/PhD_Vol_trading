"""Конфигурации стадий потока LGBM: функции потерь L1–L5, признаки F1–F6, комбинации, обратные абляции, F-all. Все на hp_y LB0 (own)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s5.b2 import engine as e2
from models_v2.s6.lgbm import cfg, grid, job, tab
from models_v2.s6.lgbm.st_lb0 import OWN
from models_v2.s6.lgbm.world import DATA, Item

OBJ = (("L1_level", "lvl", ()), ("L3_gamma", "gamma", ()), ("L4_qlike", "qlike", ()), ("L5a_rvw05", "ln", ("rvw", 0.5)),
       ("L5b_rvw10", "ln", ("rvw", 1.0)), ("L5t_decay504", "ln", ("decay", 504)))
DESC = {"L1_level": "L1 regression L2 по уровню RV/S, σ̂²=max(S·f, ε)", "L3_gamma": "L3 gamma (log-link)", "L4_qlike": "L4 custom QLIKE, F=ln σ̂²",
        "L5a_rvw05": "L5 L2 по ln-цели с весами RV_w^-0,5", "L5b_rvw10": "L5 то же, RV_w^-1", "L5t_decay504": "L5 веса по возрасту, полураспад 504 строки"}


def hp_own() -> dict:
    """{Y: Hp} собственного baseline LB0 (из сохранённого st_lb0_own.json)."""
    lab = json.loads((DATA / ("st_%s.json" % OWN)).read_text())["hp_by_year"]
    full = {h.label(): h for h in grid.full_grid()}
    return {int(Y): full[v] for Y, v in lab.items()}


def objective_items(W: dict) -> list:
    hp = hp_own()
    return [Item(n, job.base_spec(W, n, k, wt), hp, "obj_feat", "objective", DESC[n]) for n, k, wt in OBJ]


def feat_spec(W: dict, name: str, add: tuple = (), drop: str = "") -> job.Spec:
    dz = e2.make_dz(W["dz0"], W["feats"], e2.Cfg(name, add=add, drop=drop))
    return job.Spec(name, "ln", dz.X, dz.y, dz.off)


def group_cols(W: dict, groups) -> tuple:
    return tuple(c for g in groups for c in W["acc"][g])


def feat_item(W: dict, name: str, groups=(), drop: str = "", desc: str = "") -> Item:
    return Item(name, feat_spec(W, name, group_cols(W, groups), drop), hp_own(), "obj_feat", "ablation" if drop else "feature", desc)


def single_items(W: dict) -> list:
    return [feat_item(W, "F_" + g, (g,), desc="OWN + группа %s (%d столбцов)" % (g, len(W["acc"][g]))) for g in sorted(W["acc"])]


def dump_registry(path, reg: dict) -> None:
    path.write_text(json.dumps(reg, ensure_ascii=False, indent=1))


def table_rows(W: dict, names: list, path, note: str) -> pd.DataFrame:
    df = pd.DataFrame(cfg.rows(W, names, OWN))
    tab.write(df, path, note)
    return df


def top_groups(W: dict, singles: list, k: int) -> tuple:
    c = cfg.flat(singles)
    order = sorted(singles, key=lambda n: (c[n] if np.isfinite(c[n]) else np.inf))
    return tuple(n[2:] for n in order[:k])
