"""B2: внутренний критерий crit_loyo (GATES_B2 §5): годовой LOYO по целям < 01.01.Y, HP года Y, среднее годовых средних QLIKE (как s4.select)."""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

from models_v2 import metrics
from models_v2.s4 import fit, runner, select
from models_v2.s5.b2 import engine

YEARS = tuple(range(2017, 2027))


def nearest(tr: np.ndarray, val: np.ndarray, n: int = 0, hl: int = 0) -> tuple:
    """(пары, веса по строкам): n ближайших по времени к блоку val пар (обе стороны) и/или вес 0,5^(расстояние/hl)."""
    v0, v1 = val.min(), val.max()
    dist = np.where(tr < v0, v0 - tr, tr - v1).astype(float)
    if n:
        keep = np.sort(np.argsort(dist, kind="stable")[:n])
        tr, dist = tr[keep], dist[keep]
    return tr, dist


def fold(F, dz, cfg, hp, T_cut: int, year: int) -> float:
    tr, val = runner.fold_split(F, dz, T_cut, year)
    if len(val) < 50 or len(tr) < 2 * fit.MIN_ES + fit.EMBARGO:
        return float("nan")
    w = None
    if cfg.window or cfg.hl:
        tr, dist = nearest(tr, val, cfg.window, cfg.hl)
        if cfg.hl:
            w = np.zeros(len(F.rv))
            w[tr] = 0.5 ** (dist / cfg.hl)
    m = fit.fit_i1(dz.X, dz.y, tr, hp, w)
    s2 = np.exp(dz.off[val] + m["predict"](dz.X[val]) + m["s2_es"] / 2)
    return float(metrics.qlike(F.rv[val + 1], s2).mean())


def _job(a):
    name, Y, y = a
    W = engine._W
    return name, Y, y, fold(W.F, W.dzs[name], W.cfgs[name], W.i2[Y], select.cut_row(W.F, Y), y)


def run(w, names, procs: int = 4) -> dict:
    """{name: crit_loyo}; {name: {Y: годовое среднее}} в w.loyo_years не хранится — возвращается второй словарь."""
    jobs = [(n, Y, y) for n in names for Y in YEARS for y in range(select.FIRST_Y, Y)]
    res = {}
    with mp.get_context("spawn").Pool(procs, initializer=engine._init, initargs=(w,)) as p:
        for n, Y, y, q in p.imap_unordered(_job, jobs, chunksize=4):
            res.setdefault((n, Y), []).append(q)
    ann = {n: {Y: float(np.nanmean(res[(n, Y)])) for Y in YEARS} for n in names}
    return {n: float(np.mean(list(ann[n].values()))) for n in names}, ann
