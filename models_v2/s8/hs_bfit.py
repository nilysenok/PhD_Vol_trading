"""Ж-212 (HS5, G7/G8): конвейер B «HAR как признаки». XGB — HP `hp_by_year` утверждённого XJ без изменений (меняется только матрица признаков);
LGBM — собственные HP: годовой LOYO по сетке 6 точек num_leaves ∈ {4, 8, 32} × lr ∈ {0,02; 0,05} (остальное — HP C0_inh_F4 своего года, max_depth = −1,
иначе num_leaves = 32 ≡ 8 при глубине 3). Ежедневное переобучение, расширяющееся окно, seed 42; пул spawn <= 4. Нулевых добавок => побитно XJ/LF (G7)."""
from __future__ import annotations

import dataclasses
import itertools
import multiprocessing as mp

import numpy as np
import pandas as pd

from models_v2.s5.b1 import designs
from models_v2.s6.e import comp
from models_v2.s6.lgbm import c_hp, grid as lgrid, job
from models_v2.s6.lgbm import st_cfgs as sc
from models_v2.s6.xgb import c_calc, cfgx, worldx

PROCS = 4
NL, LR = (4, 8, 32), (0.02, 0.05)
POINTS = tuple(itertools.product(NL, LR))
TOL = 1e-12


def xgb_spec(w: dict, extra: np.ndarray | None) -> designs.Spec:
    sp = worldx.spec_of(w, cfgx.CTRL)
    return sp if extra is None else dataclasses.replace(sp, X=np.column_stack([sp.X, extra]))


def xgb_hpy(w: dict) -> dict:
    return c_calc.c_space.apply(c_calc.hpc_base(w), comp.x2_moves()[1])


def xgb_run(w: dict, extra: np.ndarray | None, seed: int = 42, procs: int = PROCS) -> np.ndarray:
    """σ̂² на сетке (2271) модели XGB с матрицей Б-2+F4 (+ extra)."""
    sp, g = xgb_spec(w, extra), w["grid"]
    with mp.get_context("spawn").Pool(min(procs, PROCS), initializer=c_calc.init, initargs=(w["F"], sp, xgb_hpy(w), w["eps"], seed, "l2")) as p:
        res = dict(p.imap_unordered(c_calc.block_job, [int(d) for d in g], chunksize=8))
    raw = pd.DataFrame([res[int(d)] for d in g])
    raw.insert(0, "row", g.astype(int))
    return worldx.sigma_frame(w, sp, raw)["sigma2"].to_numpy(float)


def lgbm_spec(w: dict, extra: np.ndarray | None) -> job.Spec:
    sp = sc.feat_spec(w, "hs", sc.group_cols(w, ("F4",)))
    return sp if extra is None else dataclasses.replace(sp, X=np.column_stack([sp.X, extra]))


def lgbm_inh(w: dict) -> dict:
    return c_hp.lift_y({Y: lgrid.from_i2(w["i2"][Y]) for Y in job.YEARS})


def point_hp(inh: dict, nl: int, lr: float) -> dict:
    return {Y: dataclasses.replace(h, num_leaves=nl, lr=lr, max_depth=-1) for Y, h in inh.items()}


def pick_points(folds: dict) -> dict:
    """folds {(nl, lr): {(Y, y): q}} -> {Y: (nl, lr)}: минимум среднего LOYO QLIKE по фолдам года Y; ничья (|Δ| <= 1e-12) — меньше num_leaves, затем lr."""
    out = {}
    for Y in job.YEARS:
        sc_ = {pt: float(np.nanmean([q for (yy, _), q in f.items() if yy == Y])) for pt, f in folds.items()}
        m = min(v for v in sc_.values() if np.isfinite(v))
        out[Y] = min((pt for pt, v in sc_.items() if v - m <= TOL), key=lambda pt: (pt[0], pt[1]))
    return out


def lgbm_select(w: dict, p, name: str) -> dict:
    """Годовой LOYO-выбор из POINTS на открытом пуле -> {Y: (nl, lr)}."""
    inh, f = lgbm_inh(w), {}
    for nl, lr in POINTS:
        r = job.run_folds(p, name, point_hp(inh, nl, lr))
        f[(nl, lr)] = {(k[1], k[2]): q for k, q in r.items()}
    return pick_points(f)


def lgbm_run(w: dict, extra: np.ndarray | None, own: bool, procs: int = PROCS) -> tuple:
    """-> (σ̂² на сетке, {Y: (nl, lr)} | None, hp_y). own=False — HP C0_inh_F4 без изменений (воспроизведение LF)."""
    sp, g, F = lgbm_spec(w, extra), w["grid"], w["F"]
    inh = lgbm_inh(w)
    with job.pool(F, {"hs": sp}, w["eps"], procs) as p:
        sel = lgbm_select(w, p, "hs") if own else None
        hp_y = {Y: dataclasses.replace(inh[Y], num_leaves=sel[Y][0], lr=sel[Y][1], max_depth=-1) for Y in inh} if own else inh
        res = job.run_days(p, F, "hs", hp_y, g)
    fc, _ = job.assemble(F, sp, w["eps"], g, res, "hs")
    return fc["sigma2"].to_numpy(float), sel, hp_y
