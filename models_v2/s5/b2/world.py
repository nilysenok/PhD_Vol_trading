"""B2: загрузка мира (Б-2 из s4, признаки F1–F6 с кэшем по ключу входов и кода), счёт конфигураций с кэшем и resume (Д5)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2 import prov
from models_v2.s5 import common
from models_v2.s5.b2 import cusum, engine, feat, loyo

HERE = Path(__file__).resolve().parent
OUT = common.DEEP / "b2"
BASE = engine.Cfg("BASE")


def code_sha() -> str:
    h = hashlib.sha256()
    for f in sorted(HERE.glob("*.py")):
        if f.name != "run_b2.py":
            h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def load() -> dict:
    from models_v2 import run_fix
    F, grid, DZ, _, _ = run_fix.prepare(str(common.IMOEX))
    bars, imoex = pd.read_parquet(prov.BARS), pd.read_parquet(common.IMOEX)
    key = hashlib.sha256((common.sha_file(prov.BARS) + common.sha_file(common.IMOEX) + code_sha()).encode()).hexdigest()[:16]
    p, k = OUT / "feats_b2.parquet", OUT / "feats_b2.key"
    OUT.mkdir(parents=True, exist_ok=True)
    if p.exists() and k.exists() and k.read_text() == key:
        feats = pd.read_parquet(p)
    else:
        feats = feat.build(F, bars, imoex)
        feats.to_parquet(p)
        k.write_text(key)
    return {"F": F, "grid": grid, "dz0": DZ["m_mkx"], "i2": run_fix.load_i2("i2_choice_g132"), "feats": feats, "key": key}


def make_world(L: dict, cfgs: list) -> engine.World:
    F = L["F"]
    al = {h: cusum.run(cusum.har_arrivals(F), h)[2] for h in (5.0, 8.0)}
    return engine.World(F, L["grid"], {c.name: engine.make_dz(L["dz0"], L["feats"], c) for c in cfgs}, {c.name: c for c in cfgs}, L["i2"], al)


def _paths(name: str) -> tuple:
    return OUT / ("fc_%s.parquet" % name), OUT / ("st_%s.json" % name)


def _key(L, c) -> str:
    return hashlib.sha256((c.trial_id() + L["key"] + code_sha()).encode()).hexdigest()[:16]


def compute(L: dict, cfgs: list, procs: int = 4, with_loyo: bool = True) -> None:
    """Считает недостающие конфигурации (прогнозы, статистика блоков, crit_loyo) и пишет кэш; готовые не пересчитывает."""
    todo = []
    for c in cfgs:
        fp, sp = _paths(c.name)
        ok = fp.exists() and sp.exists() and json.loads(sp.read_text()).get("key") == _key(L, c)
        if not ok:
            todo.append(c)
    if not todo:
        return
    w = make_world(L, todo)
    names = [c.name for c in todo]
    res = engine.run_all(w, names, procs)
    cl, ann = loyo.run(w, [c.name for c in todo if not (c.freq != "d" or c.cusum_h)], procs) if with_loyo else ({}, {})
    for c in todo:
        fp, sp = _paths(c.name)
        engine.forecasts(w, c.name, res[c.name]).to_parquet(fp)
        st = engine.block_stats(w, c.name, res[c.name])
        st.update(key=_key(L, c), crit_loyo=cl.get(c.name, float("nan")), loyo_years=ann.get(c.name, {}))
        sp.write_text(json.dumps(st, default=float))


def get(L, c) -> tuple:
    fp, sp = _paths(c.name)
    st = json.loads(sp.read_text())
    return pd.read_parquet(fp), st


def controls_frames() -> tuple:
    c = common.controls()
    mk = lambda col: pd.DataFrame({"d": c["d"], "target_date": c["target_date"], "sigma2": c[col], "rv": c["rv"]})  # noqa: E731
    return mk("s2_harj"), mk("s2_b2")


def check_control(fc: pd.DataFrame, b2: pd.DataFrame, tol: float = 1e-9) -> float:
    """Макс. |σ̂²/σ̂²_parquet − 1| по сетке; сетка обязана совпасть."""
    if len(fc) != common.N_GRID or not (fc["d"].to_numpy() == b2["d"].to_numpy()).all():
        raise ValueError("сетка BASE не совпала с контролем")
    return float(np.max(np.abs(fc["sigma2"].to_numpy() / b2["sigma2"].to_numpy() - 1)))
