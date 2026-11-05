"""Счёт строки волны B: ежедневный прогноз по сетке и фолды LOYO через `s5/b1/jobs` (spawn-пул <= 4). Кэш — только в data/regen/xgb_lgbm_full_v1/xgboost/,
ключ = trial_id + код (s6/xgb, jobs/fitobj/losses/designs, s4/fit) + ключ признаков (Д5)."""
from __future__ import annotations

import hashlib
import time

import pandas as pd

from models_v2 import prov
from models_v2.s5 import common
from models_v2.s6 import paths6
from models_v2.s6.xgb import cfgx, worldx

HERE = paths6.ROOT / "models_v2/s6/xgb"
DATA = paths6.OUT / "xgboost"
PROCS = 4
CODE = ("s5/b1/jobs.py", "s5/b1/fitobj.py", "s5/b1/losses.py", "s5/b1/designs.py", "s4/fit.py", "s5/b2/engine.py")


def code_sha() -> str:
    h = hashlib.sha256()
    for f in sorted(HERE.glob("*.py")):
        h.update(f.name.encode() + f.read_bytes())
    for f in CODE:
        h.update(f.encode() + (paths6.ROOT / "models_v2" / f).read_bytes())
    return h.hexdigest()[:16]


def key_of(c: dict, w: dict) -> str:
    inputs = common.sha_file(prov.BARS) + common.sha_file(common.IMOEX) + worldx.feats_key()
    return hashlib.sha256((cfgx.trial_id(c) + code_sha() + inputs).encode()).hexdigest()[:16]


def _cached(name: str, k: str, build) -> pd.DataFrame:
    p, kp = DATA / (name + ".parquet"), DATA / (name + ".key")
    if p.exists() and kp.exists() and kp.read_text() == k:
        return pd.read_parquet(p)
    df = build()
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)
    kp.write_text(k)
    return df


def raw_daily(w: dict, spec, k: str, procs: int = PROCS) -> pd.DataFrame:
    from models_v2.s5.b1 import jobs

    def build():
        res = jobs.run(w["F"], {spec.name: spec}, w["i2"], w["eps"], "block", [(spec.name, int(d)) for d in w["grid"]], procs)
        out = pd.DataFrame([res[(spec.name, int(d))] for d in w["grid"]])
        out.insert(0, "row", w["grid"].astype(int))
        return out
    return _cached("raw_" + spec.name, k, build)


def folds_loyo(w: dict, spec, k: str, procs: int = PROCS) -> pd.DataFrame:
    from models_v2.s5.b1 import jobs

    def build():
        tasks = [(spec.name, Y, y) for Y in range(2017, 2027) for y in range(2015, Y)]
        res = jobs.run(w["F"], {spec.name: spec}, w["i2"], w["eps"], "fold", tasks, procs)
        rec = [{"Y": a[1], "y": a[2], "q": qs[0]} for a, qs in res.items()]
        return pd.DataFrame(rec).sort_values(["Y", "y"]).reset_index(drop=True)
    return _cached("loyo_" + spec.name, k, build)


def evaluate(w: dict, c: dict, procs: int = PROCS) -> tuple:
    """-> (fc d/target_date/rv/sigma2, folds Y/y/q, raw-поток, секунды)."""
    spec, k, t0 = worldx.spec_of(w, c), key_of(c, w), time.time()
    raw = raw_daily(w, spec, k, procs)
    fl = folds_loyo(w, spec, k, procs)
    return worldx.sigma_frame(w, spec, raw), fl, raw, time.time() - t0
