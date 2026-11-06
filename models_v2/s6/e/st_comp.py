"""G3: сбор компонентов из сохранённых рядов и контроль против известных ответов (таблицы B/C/A)."""
from __future__ import annotations

import json

import numpy as np

from models_v2.s5.c import mix
from models_v2.s6 import paths6
from models_v2.s6.e import comp, crit_e, pe
from models_v2.s6.xgb import worldx

EXPECT = {"XJ": 0.211295, "XF": 0.212538, "B2X": 0.2205, "B2L": 0.2223}
TOL = 5e-5


def qmean(df, name, seed=42) -> float:
    return float(crit_e.qlike_series(df["rv"].to_numpy(float), comp.col(df, name, seed)).mean())


def stage(_args=None) -> int:
    w = worldx.load()
    df = comp.build(w)
    comp.save(df)
    ctx = comp.ctx_of(df)
    h3 = mix.h3("roll", ctx, comp.col(df, "XF"))
    rep = {"n": len(df), "qlike_seed42": {k: qmean(df, k) for k in ("XJ", "XF", "LF", "B2X", "B2L", "HAR")},
           "h3_roll_xf": float(crit_e.qlike_series(ctx["rv"], h3).mean()),
           "seeds": {k: [qmean(df, k, s) for s in comp.SEEDS] for k in ("XJ", "XF", "LF", "B2X", "B2L")}}
    ok = rep["n"] == paths6.N_GRID and all(abs(rep["qlike_seed42"][k] - v) < TOL for k, v in EXPECT.items())
    ok = ok and abs(rep["h3_roll_xf"] - 0.2098) < TOL and abs(rep["qlike_seed42"]["LF"] - 0.2117) < 1e-3
    pe.ENS.mkdir(parents=True, exist_ok=True)
    paths6.dump_json({**rep, "ok": bool(ok), "expect": EXPECT, "tol": TOL}, pe.ENS / "COMPONENTS_CHECK.json")
    print("G3 comp:", "OK" if ok else "ПРОВАЛ", json.dumps(rep, default=float)[:900], flush=True)
    return 0 if ok else 1
