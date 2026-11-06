"""Волна D, поток LGBM: фолды LOYO и прогнозы блоков с окном/весами (RD1, RD2). Пул — `job.pool` (spawn, <= 4 процессов)."""
from __future__ import annotations

import time

import numpy as np

from models_v2 import metrics
from models_v2.s4 import fit, rows, runner, select
from models_v2.s6.lgbm import d_win, fitl, job
from models_v2.s6.lgbm.d_win import Tm


def fold_q_d(F, sp, eps, hp, tm: Tm, Y: int, y: int):
    """(строки проверки, QLIKE по ним) фолда (Y, y): как `job.fold_q`, но обучение — ближайшие к центру года проверки win пар и веса по |t − центр|.
    Пары — с целями < 01.01.Y кроме года y; None, если мало данных."""
    T = select.cut_row(F, Y)
    tr, val = runner.fold_split(F, sp, T, y)
    if len(val) < 50 or len(tr) < 2 * fit.MIN_ES + fit.EMBARGO:
        return None
    centre = float(val.mean())
    idx = d_win.nearest(tr, centre, tm.win)
    m = fitl.fit_l(sp.X, sp.y, idx, hp, sp.kind, d_win.weights_full(len(F.rv), centre, tm.hl))
    s2, _ = fitl.sigma2(sp.kind, val, m["predict"](sp.X[val]), m["s2_es"], sp.off, np.exp(sp.off), eps)
    return val, np.asarray(metrics.qlike(F.rv[val + 1], s2), float)


def block_fc(F, sp, hp, tm: Tm, anchor: int, ds: list) -> dict:
    """Одно обучение на якоре (пары t+1 <= anchor) и прогноз всех дат блока ds своими X[d]. -> {d: поля как `job.day_fc`}."""
    t0 = time.time()
    idx = d_win.nearest(job.pairs(F, sp, anchor), float(anchor), tm.win)
    m = fitl.fit_l(sp.X, sp.y, idx, hp, sp.kind, d_win.weights_full(len(F.rv), float(anchor), tm.hl))
    f = m["predict"](sp.X[ds])
    secs = time.time() - t0
    return {int(d): {"f": float(fi), "s2": m["s2_es"], "k": m["k"], "k_cap": m["k_cap"], "nf": m["n_nonfinite_es"], "secs": secs if i == 0 else 0.0}
            for i, (d, fi) in enumerate(zip(ds, f))}


def _fold_job(a):
    name, hp, tm, Y, y = a
    return (Y, y), fold_q_d(job._S["F"], job._S["specs"][name], job._S["eps"], hp, tm, Y, y)


def _block_job(a):
    name, hp, tm, anchor, ds = a
    return block_fc(job._S["F"], job._S["specs"][name], hp, tm, anchor, ds)


def run_folds(p, name: str, hp_y: dict, tm: Tm) -> dict:
    """{(Y, y): (val, q) | None}."""
    return dict(p.imap_unordered(_fold_job, [(name, hp_y[Y], tm, Y, y) for Y, y in job.FOLDS], chunksize=2))


def run_blocks(p, F, name: str, hp_y: dict, tm: Tm, blocks: list) -> dict:
    """{(name, d): прогноз} для `job.assemble`; HP — года якоря."""
    out = {}
    for r in p.imap_unordered(_block_job, [(name, hp_y[rows.year_of(F, a)], tm, a, ds) for a, ds in blocks], chunksize=4):
        out.update({(name, d): v for d, v in r.items()})
    return out
