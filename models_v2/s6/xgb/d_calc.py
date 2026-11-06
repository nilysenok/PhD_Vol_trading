"""Счёт строки волны D: блоки прогноза (ежедневно / еженедельно / ежемесячно, окно, веса) и фолды LOYO с потерями по датам. spawn-пул <= 4.
Кэш — `data/regen/xgb_lgbm_full_v1/xgboost/d/`; ключ = trial_id + код (`s6/xgb/d_*`, designs, jobs, fit, fitx, loyo) + хэш входов + HP по годам (Д5)."""
from __future__ import annotations

import contextlib
import hashlib
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s4 import fit, rows, select
from models_v2.s5.b1 import designs, jobs
from models_v2.s5.c import fitx
from models_v2.s6 import paths6
from models_v2.s6.xgb import c_calc, d_fit, d_space, worldx

HERE = paths6.ROOT / "models_v2/s6/xgb"
DATA = paths6.OUT / "xgboost" / "d"
PROCS = 4
CODE = ("s5/b1/designs.py", "s5/b1/jobs.py", "s4/fit.py", "s5/c/fitx.py", "s4/rows.py", "s4/select.py", "s5/b2/loyo.py", "s5/b2/engine.py")
_S: dict = {}


def init(F, spec, hpy, eps, win, hl):
    _S.update(F=F, spec=spec, hpy=hpy, eps=eps, win=win, hl=hl)


def block_job(a):
    T, ds = a
    F, spec, t0 = _S["F"], _S["spec"], time.time()
    m = d_fit.fit_d(spec.X, spec.y, jobs.pairs(F, spec, T), _S["hpy"][rows.year_of(F, T)], _S["win"], _S["hl"], T)
    f = np.asarray(m["predict"](spec.X[np.asarray(ds)]), float)
    return T, ds, f, m["s2_es"], m["k"], time.time() - t0


def fold_job(a):
    """-> (Y, y), (строки val, потери QLIKE по датам) или None, если фолд не определён."""
    Y, y = a
    F, spec = _S["F"], _S["spec"]
    idx = jobs.pairs(F, spec, select.cut_row(F, Y))
    yr = F.dates[idx + 1].astype("datetime64[Y]").astype(int) + 1970
    tr, val = idx[yr != y], idx[yr == y]
    if len(val) < 50 or len(tr) < 2 * fit.MIN_ES + fit.EMBARGO:
        return a, None
    tr, w = d_fit.fold_train(tr, val, _S["win"], _S["hl"], len(spec.y))
    m = fitx.fit_i1x(spec.X, spec.y, tr, _S["hpy"][Y], w, None, fit.SEED)
    s2 = designs.sigma2(spec, val, m["predict"](spec.X[val]), m["s2_es"], _S["eps"], spec.cs[0])[0]
    return a, (val, np.asarray(metrics.qlike(F.rv[val + 1], s2), float))


def code_sha() -> str:
    h = hashlib.sha256()
    for f in sorted(HERE.glob("d_*.py")):
        h.update(f.name.encode() + f.read_bytes())
    for f in CODE:
        h.update(f.encode() + (paths6.ROOT / "models_v2" / f).read_bytes())
    return h.hexdigest()[:16]


def _cached(name: str, k: str, build) -> pd.DataFrame:
    p, kp = DATA / (name + ".parquet"), DATA / (name + ".key")
    if p.exists() and kp.exists() and kp.read_text() == k:
        return pd.read_parquet(p)
    df = build()
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)
    kp.write_text(k)
    return df


def folds_mean(long: pd.DataFrame) -> pd.DataFrame:
    """Фолды Y, y, q из потерь по датам (q — средний QLIKE фолда)."""
    return long.groupby(["Y", "y"], as_index=False)["q"].mean().sort_values(["Y", "y"]).reset_index(drop=True)


def evaluate(w: dict, cfg: d_space.DCfg, daily: bool = True, folds: bool = True, procs: int = PROCS) -> dict:
    """-> {'tid', 'name', 'fc' (d/target_date/rv/sigma2), 'raw' (row/f/s2/k/T/secs), 'long' (Y/y/row/q), 'folds' (Y/y/q), 'secs'}."""
    tid, name, t0 = d_space.trial_id(cfg), d_space.name_of(cfg), time.time()
    spec, hpy, (win, hl) = d_space.spec_for(w, cfg.r1), d_space.hpy_of(w, cfg.base), d_space.WINDOWS[cfg.win]
    k = hashlib.sha256((tid + code_sha() + c_calc.inputs_sha() + repr(sorted((Y, repr(h)) for Y, h in hpy.items()))).encode()).hexdigest()[:16]
    out, box = {"tid": tid, "name": name}, {}
    with contextlib.ExitStack() as stack:
        def pool():      # пул создаётся, только если кэш не подошёл
            if "p" not in box:
                box["p"] = stack.enter_context(mp.get_context("spawn").Pool(min(procs, PROCS), initializer=init, initargs=(w["F"], spec, hpy, w["eps"], win, hl)))
            return box["p"]

        def b_raw():
            rec = {}
            for T, ds, f, s2, kk, sec in pool().imap_unordered(block_job, list(d_space.blocks(w["F"], w["grid"], cfg.freq).items()), chunksize=4):
                rec.update({int(d): (float(fv), s2, kk, T, sec / len(ds)) for d, fv in zip(ds, f)})
            return pd.DataFrame([(int(d), *rec[int(d)]) for d in w["grid"]], columns=["row", "f", "s2", "k", "T", "secs"])

        def b_long():
            tasks = [(Y, y) for Y in paths6.YEARS for y in range(select.FIRST_Y, Y)]
            res = dict(pool().imap_unordered(fold_job, tasks, chunksize=2))
            return pd.DataFrame([(a[0], a[1], int(r), float(q)) for a, v in sorted(res.items()) if v is not None for r, q in zip(*v)], columns=["Y", "y", "row", "q"])
        if daily:
            out["raw"] = _cached("raw_" + name, k, b_raw)
            out["fc"] = worldx.sigma_frame(w, spec, out["raw"])
        if folds:
            out["long"] = _cached("foldsq_" + name, k, b_long)
            out["folds"] = folds_mean(out["long"])
    out["secs"] = time.time() - t0
    return out
