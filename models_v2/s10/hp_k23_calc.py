"""P1a: счёт K2 (XGB, `x2_best_j` = И2 + сдвиги из X2_DECISION.json) и K3 (LGBM, `C0_inh_F4` = HP И2 → LightGBM) на строках сетки бумаги.
Обучение — замороженные функции без изменений: `c_calc.block_job` (K2) и `job.run_days` / `job.assemble` (K3): ежедневное переобучение на расширяющемся окне, seed 42, ранняя остановка И1.
Свои — только пул процессов (<= 2, spawn) и подбор строк. Новой оптимизации нет."""
from __future__ import annotations

import json
import multiprocessing as mp

import numpy as np
import pandas as pd

from models_v2.s4 import rows as R
from models_v2.s6.lgbm import c_hp, c_l0, grid, job
from models_v2.s6.xgb import c_calc, c_space, cfgx, worldx
from models_v2.s10 import hp_k23_world as KW

SEED = 42


def x2_moves() -> dict:
    """Замороженные сдвиги `x2_best_j` (X2_DECISION.json, Ж-206)."""
    return json.loads((cfgx.XGB_REP / "X2_DECISION.json").read_text())["best_moves"]


def year_of_rows(w: dict, rows) -> np.ndarray:
    return np.array([R.year_of(w["F"], int(d)) for d in rows], int)


def select_rows(w: dict, y0, dates=None) -> np.ndarray:
    """Строки сетки `har.eligible`, цель которых в годе >= y0 (У-3) и (если задано) дата d входит в `dates` панели; y0 None — пусто."""
    g = w["grid"]
    if y0 is None or len(g) == 0:
        return np.array([], int)
    keep = year_of_rows(w, g) >= y0
    if dates is not None:
        keep &= np.isin(w["F"].dates[g].astype("datetime64[ns]"), np.asarray(dates, dtype="datetime64[ns]"))
    return g[keep]


def hpc_label(h) -> str:
    return "d%d mc%g l2 %g lr%g ss%g cs%g g%g a%g mds%g" % (h.depth, h.min_child, h.l2, h.lr, h.subsample, h.colsample, h.gamma, h.alpha, h.mds)


def k2_sigma(w: dict, i2: dict, rows, procs: int = 2) -> tuple:
    """-> (σ̂² K2 по rows, {Y: HPC}). Каждому Y из rows нужен i2[Y]."""
    spec = worldx.spec_of(w, cfgx.CTRL)
    hpy = c_space.apply({Y: c_space.to_hpc(h) for Y, h in i2.items()}, x2_moves())
    rows = np.asarray(rows, int)
    if len(rows) == 0:
        return np.array([]), hpy
    with mp.get_context("spawn").Pool(min(procs, 2), initializer=c_calc.init, initargs=(w["F"], spec, hpy, w["eps"], SEED, "l2")) as p:
        res = dict(p.imap_unordered(c_calc.block_job, [int(d) for d in rows], chunksize=8))
    raw = pd.DataFrame([res[int(d)] for d in rows])
    raw.insert(0, "row", rows)
    return worldx.sigma_frame(w, spec, raw)["sigma2"].to_numpy(float), hpy


def k3_sigma(w: dict, i2: dict, rows, procs: int = 2) -> tuple:
    """-> (σ̂² K3 по rows, {Y: Hp}). Спецификация — F4 (c_l0.spec_of), HP — наследованные И2 (`grid.from_i2`)."""
    spec = c_l0.spec_of(w, "K3", "F4")
    hp_y = c_hp.lift_y({Y: grid.from_i2(h) for Y, h in i2.items()})
    rows = np.asarray(rows, int)
    if len(rows) == 0:
        return np.array([]), hp_y
    with job.pool(w["F"], {"K3": spec}, w["eps"], min(procs, 2)) as p:
        res = job.run_days(p, w["F"], "K3", hp_y, rows)
    fc, _ = job.assemble(w["F"], spec, w["eps"], rows, res, "K3")
    return fc["sigma2"].to_numpy(float), hp_y


def forecast_frame(w: dict, i2: dict, rows, procs: int = 2) -> pd.DataFrame:
    """Строки rows -> d, t_plus, rv, sigma2_k2, sigma2_k3, hp_year, hp_k2, hp_k3 (боевой путь для тестов и счёта)."""
    rows = np.asarray(rows, int)
    s2x, hpx = k2_sigma(w, i2, rows, procs)
    s2l, hpl = k3_sigma(w, i2, rows, procs)
    yr = year_of_rows(w, rows)
    F = w["F"]
    return pd.DataFrame({"date": F.dates[rows], "t_plus": F.dates[rows + 1], "rv": F.rv[rows + 1], "sigma2_k2": s2x, "sigma2_k3": s2l, "hp_year": yr,
                         "hp_k2": [hpc_label(hpx[y]) for y in yr], "hp_k3": [hpl[y].label() for y in yr]})
