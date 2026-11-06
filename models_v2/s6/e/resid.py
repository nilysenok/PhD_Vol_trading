"""H4/H5 (GATES_E R5, G5): HAR-J(б) с остаточным бустингом. Цель e_t = ln RV_{t+1} − lb_t (lb — PIT HAR-J(б) по данным <= t, `resid.harj_ab`), σ̂² = exp(lb + f + s²/2).
Признаки: Б-1 (mkx, форма resid) + F4; HP годового выбора И2 (XGB — как ctrl_f4, LGBM — наследованные, как C0_inh_F4); ежедневное переобучение, расширяющееся окно."""
from __future__ import annotations

import json
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

from models_v2 import run_fix
from models_v2.s5 import common, pit
from models_v2.s5.b1 import designs
from models_v2.s5.b2 import engine
from models_v2.s6 import paths6
from models_v2.s6.e import comp, pe
from models_v2.s6.lgbm import c_hp, grid as lgrid, job, world as lworld
from models_v2.s6.xgb import c_calc, c_space

B2DIR = paths6.OLD_DATA[1] / "b2"


def build_world() -> dict:
    F, grid, DZ, _, _ = run_fix.prepare(str(common.IMOEX))
    acc = {g: tuple(c) for g, c in json.loads((B2DIR / "accepted.json").read_text()).items()}
    feats = pd.read_parquet(B2DIR / "feats_b2.parquet")
    dz = engine.make_dz(DZ["resid_mkx"], feats, engine.Cfg("h45", add=acc["F4"]))
    return {"F": F, "grid": np.asarray(grid, int), "dz": dz, "eps": pit.eps_floor(F.rv), "i2": run_fix.load_i2("i2_choice_g132"), "ctrl": common.controls()}


def lb_check(W: dict) -> float:
    """Контроль G5: exp(lb) на сетке = s2_harj из controls (макс. отн. разность)."""
    lb = W["dz"].off[W["grid"]]
    return float(np.max(np.abs(np.exp(lb) / W["ctrl"]["s2_harj"].to_numpy(float) - 1.0)))


def _frame(W: dict, s2: np.ndarray) -> pd.DataFrame:
    F, i = W["F"], W["grid"]
    return pd.DataFrame({"d": F.dates[i], "target_date": F.dates[i + 1], "rv": F.rv[i + 1], "sigma2": s2})


def run_h4(W: dict, seed: int = 42, procs: int = 4) -> pd.DataFrame:
    F, dz = W["F"], W["dz"]
    spec = designs.Spec("h4", "l2", dz.X, dz.y, "ln", dz.off)
    hpy = {Y: c_space.to_hpc(W["i2"][Y]) for Y in paths6.YEARS}
    with mp.get_context("spawn").Pool(procs, initializer=c_calc.init, initargs=(F, spec, hpy, W["eps"], seed, "l2")) as p:
        res = dict(p.imap_unordered(c_calc.block_job, [int(d) for d in W["grid"]], chunksize=8))
    f, s2e = np.array([res[int(d)]["f"] for d in W["grid"]]), np.array([res[int(d)]["s2"] for d in W["grid"]])
    return _frame(W, designs.sigma2(spec, W["grid"], f, s2e, W["eps"])[0])


def run_h5(W: dict, seed: int = 42, procs: int = 4) -> pd.DataFrame:
    F, dz = W["F"], W["dz"]
    spec = job.Spec("h5", "ln", dz.X, dz.y, dz.off)
    hp_y = lworld.with_hp(c_hp.lift_y({Y: lgrid.from_i2(W["i2"][Y]) for Y in job.YEARS}), seed=seed)
    with job.pool(F, {"h5": spec}, W["eps"], procs) as p:
        res = job.run_days(p, F, "h5", hp_y, W["grid"])
    fc, _st = job.assemble(F, spec, W["eps"], W["grid"], res, "h5")
    return _frame(W, fc["sigma2"].to_numpy(float))


def path(kind: str, seed: int):
    return pe.OUT / ("resid_%s_s%d.parquet" % (kind, seed))


def stage(a=None) -> int:
    W = build_world()
    err = lb_check(W)
    ok = err < 1e-9
    print("G5 lb: max отн. разность exp(lb) и s2_harj = %.3g" % err, flush=True)
    for kind, fn in (("h4", run_h4), ("h5", run_h5)):
        if ok and not path(kind, 42).exists():
            t0 = time.time()
            fn(W).to_parquet(path(kind, 42))
            print("G5 %s готово за %.0f с" % (kind, time.time() - t0), flush=True)
    paths6.dump_json({"lb_max_rel_diff": err, "ok": bool(ok)}, pe.ENS / "RESID_CHECK.json")
    return 0 if ok else 1


def seed_series(kind: str, seed: int) -> pd.DataFrame:
    """Seed 43–46 для финалистов (контроль, не trial); кэш по файлу."""
    if not path(kind, seed).exists():
        W = build_world()
        (run_h4 if kind == "h4" else run_h5)(W, seed).to_parquet(path(kind, seed))
    return pd.read_parquet(path(kind, seed))
