"""Счёт строки волны C: ежедневный прогноз (расширяющееся окно, ежедневное переобучение) и фолды LOYO (HP года Y) с годовыми HPC и seed. spawn-пул <= 4.
Кэш — только `data/regen/xgb_lgbm_full_v1/xgboost/c/`; ключ = trial_id + seed + код (`s6/xgb/c_*`, designs, jobs, fit, fitx) + хэш входов (Д5)."""
from __future__ import annotations

import functools
import hashlib
import multiprocessing as mp
import time

import pandas as pd

from models_v2 import metrics, prov
from models_v2.s4 import fit, rows, select
from models_v2.s5 import common
from models_v2.s5.b1 import designs, jobs
from models_v2.s6 import paths6
from models_v2.s6.xgb import c_fit, c_space, cfgx, worldx

HERE = paths6.ROOT / "models_v2/s6/xgb"
DATA = paths6.OUT / "xgboost" / "c"
PROCS = 4
CODE = ("s5/b1/designs.py", "s5/b1/jobs.py", "s4/fit.py", "s5/c/fitx.py", "s4/rows.py", "s4/select.py")
SPEC_CFG = {"l2": cfgx.CTRL, "gamma": cfgx.NEW[0]}
_S: dict = {}


def init(F, spec, hpy, eps, seed, loss):
    _S.update(F=F, spec=spec, hpy=hpy, eps=eps, seed=seed, loss=loss)


def block_job(d):
    F, spec = _S["F"], _S["spec"]
    m = c_fit.fit_c(spec.X, spec.y, jobs.pairs(F, spec, d), _S["hpy"][rows.year_of(F, d)], _S["loss"], _S["seed"])
    return d, {"f": float(m["predict"](spec.X[[d]])[0]), "s2": m["s2_es"], "k": m["k"]}


def fold_job(a):
    Y, y = a
    F, spec = _S["F"], _S["spec"]
    idx = jobs.pairs(F, spec, select.cut_row(F, Y))
    yr = F.dates[idx + 1].astype("datetime64[Y]").astype(int) + 1970
    tr, val = idx[yr != y], idx[yr == y]
    if len(val) < 50 or len(tr) < 2 * fit.MIN_ES + fit.EMBARGO:
        return a, float("nan")
    m = c_fit.fit_c(spec.X, spec.y, tr, _S["hpy"][Y], _S["loss"], _S["seed"])
    s2 = designs.sigma2(spec, val, m["predict"](spec.X[val]), m["s2_es"], _S["eps"], spec.cs[0])[0]
    return a, float(metrics.qlike(F.rv[val + 1], s2).mean())


@functools.lru_cache(maxsize=1)
def inputs_sha() -> str:
    return common.sha_file(prov.BARS) + common.sha_file(common.IMOEX) + worldx.feats_key()


def code_sha() -> str:
    h = hashlib.sha256()
    for f in sorted(HERE.glob("c_*.py")):
        h.update(f.name.encode() + f.read_bytes())
    for f in CODE:
        h.update(f.encode() + (paths6.ROOT / "models_v2" / f).read_bytes())
    return h.hexdigest()[:16]


def spec_for(w: dict, family: str):
    return worldx.spec_of(w, SPEC_CFG[family])


def hpc_base(w: dict) -> dict:
    """{Y: HPC} — годовой выбор И2 (чтение), годы 2017…2026."""
    return {Y: c_space.to_hpc(w["i2"][Y]) for Y in paths6.YEARS}


def _cached(name: str, k: str, build) -> pd.DataFrame:
    p, kp = DATA / (name + ".parquet"), DATA / (name + ".key")
    if p.exists() and kp.exists() and kp.read_text() == k:
        return pd.read_parquet(p)
    df = build()
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)
    kp.write_text(k)
    return df


def evaluate(w: dict, family: str, moves: dict, seed: int = 42, daily: bool = True, folds: bool = True, procs: int = PROCS,
             hpy: dict | None = None, tid: str | None = None) -> dict:
    """-> {'fc': d/target_date/rv/sigma2, 'folds': Y/y/q, 'secs'}; нужное отключается флагами (seed != 42: обычно только daily).
    hpy/tid — годовой набор HP, не сводимый к одному сдвигу (годовой выбор X2); tid тогда обязателен и несёт хэш выбора."""
    tid, t0 = tid or c_space.trial_id(family, moves), time.time()
    spec, hpy, out = spec_for(w, family), hpy or c_space.apply(hpc_base(w), moves), {"tid": tid}
    k = hashlib.sha256((tid + str(seed) + code_sha() + inputs_sha() + repr(sorted((Y, repr(h)) for Y, h in hpy.items()))).encode()).hexdigest()[:16]
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(procs, PROCS), initializer=init, initargs=(w["F"], spec, hpy, w["eps"], seed, family)) as p:
        if daily:
            def b_daily():
                res = dict(p.imap_unordered(block_job, [int(d) for d in w["grid"]], chunksize=8))
                o = pd.DataFrame([res[int(d)] for d in w["grid"]])
                o.insert(0, "row", w["grid"].astype(int))
                return o
            raw = _cached("raw_%s_s%d" % (tid, seed), k, b_daily)
            out["fc"] = worldx.sigma_frame(w, spec, raw)
        if folds:
            def b_folds():
                res = dict(p.imap_unordered(fold_job, [(Y, y) for Y in paths6.YEARS for y in range(select.FIRST_Y, Y)], chunksize=4))
                return pd.DataFrame([{"Y": a[0], "y": a[1], "q": q} for a, q in res.items()]).sort_values(["Y", "y"]).reset_index(drop=True)
            out["folds"] = _cached("loyo_%s_s%d" % (tid, seed), k, b_folds)
    out["secs"] = time.time() - t0
    return out


def hp_example(w: dict, moves: dict) -> str:
    """HP одного года (2023) после сдвига — для таблицы осей."""
    h = c_space.apply(hpc_base(w), moves)[2023]
    return "d%d mc%g l2 %g lr%g ss%g cs%g g%g a%g mds%g" % (h.depth, h.min_child, h.l2, h.lr, h.subsample, h.colsample, h.gamma, h.alpha, h.mds)
