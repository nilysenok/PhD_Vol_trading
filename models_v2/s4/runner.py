"""Параллельные задачи Ж-200 (spawn, nthread=1 в каждой): фолды LOYO и блоки прогноза."""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

from models_v2 import metrics
from models_v2.s4 import fit

_S = {}


def _init(F, DZ):
    _S.update(F=F, DZ=DZ)


def pairs_upto(dz, F, T):
    return fit.pair_index(F.seg, dz.X, dz.y, (0, 1, 2, 3), T)


def fold_split(F, dz, T_cut, year) -> tuple:
    """(обучение, проверка): пары с t+1 <= T_cut; проверка — цели года `year`, обучение — все остальные."""
    idx = pairs_upto(dz, F, T_cut)
    yr = F.dates[idx + 1].astype("datetime64[Y]").astype(int) + 1970
    return idx[yr != year], idx[yr == year]


def fold_score(F, dz, hp, T_cut, year, wflag) -> float:
    """QLIKE (б) на целях года `year` модели (И1), обученной на парах с целями < 01.01.Y кроме года `year`."""
    tr, val = fold_split(F, dz, T_cut, year)
    if len(val) < 50 or len(tr) < 2 * fit.MIN_ES + fit.EMBARGO:
        return float("nan")
    w = fit.weights(np.arange(len(F.rv)), T_cut) if wflag else None
    m = fit.fit_i1(dz.X, dz.y, tr, hp, w)
    s2 = np.exp(dz.off[val] + m["predict"](dz.X[val]) + m["s2_es"] / 2)
    return float(metrics.qlike(F.rv[val + 1], s2).mean())


def _fold_job(a):
    key, dzk, hp, T_cut, year, wflag = a
    return key, fold_score(_S["F"], _S["DZ"][dzk], hp, T_cut, year, wflag)


def block_fit(F, dz, hp, T, d_idx, wflag, mono, variant):
    idx = pairs_upto(dz, F, T)
    X, d = dz.X, np.asarray(d_idx)
    if variant == "oracle":   # (i) цель как признак
        X = np.column_stack([X, dz.y])
    elif variant == "perm":   # (iii) строки X перемешаны внутри окна
        X = X.copy()
        X[idx] = X[idx][np.random.default_rng(fit.SEED + int(T)).permutation(len(idx))]
    w = fit.weights(np.arange(len(F.rv)), T) if wflag else None
    m = fit.fit_i1(X, dz.y, idx, hp, w, mono if variant != "oracle" or mono is None else tuple(mono) + (0,))
    return {"f": m["predict"](X[d]), "k": m["k"], "s2_es": m["s2_es"], "s2_tr": m["s2_tr"], "s2_tr_prelim": m["s2_tr_prelim"],
            "n_train": m["n_train"], "shrunk": m["shrunk"], "last_t": int(idx.max()), "min_t": int(idx.min())}


def _block_job(a):
    key, dzk, hp, T, d_idx, wflag, mono, variant = a
    return key, block_fit(_S["F"], _S["DZ"][dzk], hp, T, d_idx, wflag, mono, variant)


def map_jobs(F, DZ, jobs, kind: str, procs: int = 4) -> list:
    fn = _fold_job if kind == "fold" else _block_job
    with mp.get_context("spawn").Pool(procs, initializer=_init, initargs=(F, DZ)) as p:
        return list(p.imap_unordered(fn, jobs, chunksize=4))
