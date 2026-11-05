"""Задачи B1: блок прогноза на дату d (ежедневное переобучение), фолд LOYO года y при HP года Y, эксперты режимов (R2). spawn-пул <= 4."""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

from models_v2 import metrics
from models_v2.s4 import fit, rows, select
from models_v2.s5.b1 import designs, fitobj

_S: dict = {}
MAX_PROCS = 4


def init(F, specs, i2, eps):
    _S.update(F=F, specs=specs, i2=i2, eps=eps)


def hp_at(F, i2, d: int):
    return i2[rows.year_of(F, d)]


def fit_at(F, spec, idx, hp):
    w = designs.weights_l5(F, spec.gw, idx) if spec.gw else None
    return fitobj.fit_loss(spec.X, spec.y, idx, hp, spec.loss, w)


def pairs(F, spec, T: int) -> np.ndarray:
    return fit.pair_index(F.seg, spec.X, spec.y, (0, 1, 2, 3), T)


def block_job(a):
    name, d = a
    F, spec = _S["F"], _S["specs"][name]
    m = fit_at(F, spec, pairs(F, spec, d), hp_at(F, _S["i2"], d))
    return (name, d), {"f": float(m["predict"](spec.X[[d]])[0]), "s2": m["s2_es"], "k": m["k"], "tr_loss": m["tr_loss"],
                       "es_loss": m["es_loss"], "n_train": m["n_train"]}


def fold_job(a):
    name, Y, y = a
    F, spec = _S["F"], _S["specs"][name]
    idx = pairs(F, spec, select.cut_row(F, Y))
    yr = F.dates[idx + 1].astype("datetime64[Y]").astype(int) + 1970
    tr, val = idx[yr != y], idx[yr == y]
    if len(val) < 50 or len(tr) < 2 * fit.MIN_ES + fit.EMBARGO:
        return a, [float("nan")] * len(spec.cs)
    m = fit_at(F, spec, tr, select_hp(Y))
    f = m["predict"](spec.X[val])
    q = [float(metrics.qlike(F.rv[val + 1], designs.sigma2(spec, val, f, m["s2_es"], _S["eps"], c)[0]).mean()) for c in spec.cs]
    return a, q


def select_hp(Y: int):
    return _S["i2"][Y]


def experts_job(a):
    """R2: эксперт группы k обучен на парах окна с группой строки t == k; пар < N_MIN — NaN (резерв — Б-2)."""
    name, d = a
    F, spec, hp = _S["F"], _S["specs"][name], hp_at(_S["F"], _S["i2"], a[1])
    idx, out = pairs(F, spec, d), []
    for k in range(spec.ng):
        ik = idx[spec.grp[idx] == k]
        if len(ik) < designs.N_MIN:
            out.append((float("nan"), float("nan"), len(ik)))
            continue
        m = fit_at(F, spec, ik, hp)
        out.append((float(m["predict"](spec.X[[d]])[0]), m["s2_es"], len(ik)))
    return a, out


FN = {"block": block_job, "fold": fold_job, "experts": experts_job}


def run(F, specs, i2, eps, kind: str, tasks: list, procs: int = MAX_PROCS) -> dict:
    procs = min(procs, MAX_PROCS)
    with mp.get_context("spawn").Pool(procs, initializer=init, initargs=(F, specs, i2, eps)) as p:
        return dict(p.imap_unordered(FN[kind], tasks, chunksize=8))
