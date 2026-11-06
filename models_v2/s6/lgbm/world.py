"""Мир потока LGBM: F, сетка, дизайн Б-2, признаки F1–F6 (читаются из кэша B2 Ж-204, без записи), HAR-J(б), baseline, eps; кэш элементов (Д5)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s5 import common, pit
from models_v2.s6 import paths6
from models_v2.s6.lgbm import job, tab
from models_v2.s6.lgbm.grid import Hp, trial_id

HERE = Path(__file__).resolve().parent
B2DIR = paths6.OLD_DATA[1] / "b2"
DATA = tab.DATA


def code_sha() -> str:
    h = hashlib.sha256()
    for f in sorted(HERE.glob("*.py")):
        if f.name not in ("run_b.py", "print_grid.py"):
            h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def load() -> dict:
    L = paths6.world()
    F = L["F"]
    feats = pd.read_parquet(B2DIR / "feats_b2.parquet")
    if not (pd.DatetimeIndex(feats.index).to_numpy("datetime64[ns]") == F.dates.astype("datetime64[ns]")).all():
        raise ValueError("индекс признаков F1–F6 не равен датам F")
    L.update(feats=feats, acc={g: tuple(c) for g, c in json.loads((B2DIR / "accepted.json").read_text()).items()},
             eps=pit.eps_floor(F.rv), har=common.controls()[["d", "target_date", "rv", "s2_harj"]].rename(columns={"s2_harj": "sigma2"}))
    return L


@dataclass
class Item:
    name: str
    spec: job.Spec
    hp_y: dict
    direction: str            # obj_feat | hp
    kind: str                 # описание: objective / feature / ablation / arch / control
    desc: str = ""
    folds: bool = True
    seed: int = 42

    def tid(self) -> str:
        hp = {str(Y): h.label() for Y, h in sorted(self.hp_y.items())}
        feats = [self.spec.X.shape[1], hashlib.sha256(np.ascontiguousarray(self.spec.X).tobytes()).hexdigest()[:12]]
        return trial_id(self.spec.kind, hp, feats, "", {"wt": list(self.spec.wt), "name": self.name, "seed": self.seed})

    def key(self) -> str:
        return hashlib.sha256((self.tid() + code_sha()).encode()).hexdigest()[:16]


def paths(name: str) -> tuple:
    return DATA / ("fc_%s.parquet" % name), DATA / ("st_%s.json" % name)


def cached(it: Item) -> bool:
    fp, sp = paths(it.name)
    return fp.exists() and sp.exists() and json.loads(sp.read_text()).get("key") == it.key()


def with_hp(hp_y: dict, **kw) -> dict:
    return {Y: replace(h, **kw) for Y, h in hp_y.items()}
