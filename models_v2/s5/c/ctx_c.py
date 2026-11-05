"""C: загрузка мира (B2: F, сетка, Б-2, признаки), кэш ежедневных прогнозов по ключу (входы + код), сохранение строк волны."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s5 import common
from models_v2.s5.b2 import engine, world as b2w
from models_v2.s5.c import engine_c

HERE = Path(__file__).resolve().parent
DATA = common.DEEP / "c"
FILES = ("fitx.py", "space.py", "engine_c.py", "tpe.py", "mix.py", "tab_c.py", "robust.py", "h2l4.py", "ctx_c.py")
DZ_CFG = {"f4": ("F4",), "f4_range": ("f4_range",), "f4_park": ("f4_park",), "f4_f1": ("F4", "F1"), "top2": ("F4", "F3")}


def code_sha() -> str:
    h = hashlib.sha256(b2w.code_sha().encode())
    for f in FILES:
        h.update(f.encode() + (HERE / f).read_bytes())
    return h.hexdigest()[:16]


def to_hpx(h):
    from models_v2.s5.c.fitx import HPX
    return HPX(h.depth, h.min_child, h.l2, h.lr, h.subsample, h.colsample, h.max_trees, getattr(h, "gamma", 0.0), getattr(h, "alpha", 0.0))


def load() -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    L = b2w.load()
    acc = json.loads((b2w.OUT / "accepted.json").read_text())
    if len(acc["F4"]) != 5:
        raise SystemExit("стоп: принятых F4 не 5 (%d)" % len(acc["F4"]))
    cols = {k: tuple(c for g in v for c in (acc[g] if g in acc else (g,))) for k, v in DZ_CFG.items()}
    dzs = {k: engine.make_dz(L["dz0"], L["feats"], engine.Cfg("c_" + k, add=c)) for k, c in cols.items()}
    w = engine_c.W(L["F"], L["grid"], dzs)
    return {"L": L, "w": w, "ctrl": common.controls(), "i2": {Y: to_hpx(h) for Y, h in L["i2"].items()}, "cols": cols}


def cached_daily(p, w, name: str, hp_y: dict, seed: int, key0: str) -> tuple:
    k = hashlib.sha256((key0 + code_sha() + name + engine_c.hp_key(hp_y, seed)).encode()).hexdigest()[:14]
    fp, sp = DATA / ("fc_%s_%s.parquet" % (name, k)), DATA / ("st_%s_%s.json" % (name, k))
    if fp.exists() and sp.exists():
        return pd.read_parquet(fp), json.loads(sp.read_text())
    fc, st = engine_c.daily(p, w, name, hp_y, seed)
    fc.to_parquet(fp)
    sp.write_text(json.dumps(st))
    return fc, st


def load_json(name: str, default):
    f = DATA / name
    return json.loads(f.read_text()) if f.exists() else default


def save_json(name: str, obj) -> None:
    (DATA / name).write_text(json.dumps(obj, ensure_ascii=False, default=float))


def put_row(rid: str, row: dict, trial: dict | None) -> None:
    rows = load_json("rows.json", {})
    rows[rid] = row
    save_json("rows.json", rows)
    if trial is not None:
        tr = load_json("trials_rows.json", {})
        tr[trial["trial_id"]] = trial
        save_json("trials_rows.json", tr)
