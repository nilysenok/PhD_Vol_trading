# -*- coding: utf-8 -*-
"""Zh-190 stage 1: jobs that produce prediction cells (one cell = h, cutoff year, configuration); pool of 8 processes, fork."""
import json
import multiprocessing as mp
import time
import numpy as np
import pandas as pd
import gl3_zh190_boost_core as K
import gl3_zh190_boost_eval as E
import gl3_zh190_boost_schemes as S

G = {}  # data and engine, loaded before the fork
SEEDS = tuple(range(42, 52))


def cid(scheme, loss, opt=False, bag=False):
    return f'{scheme}-{loss}' + ('-opt' if opt else '') + ('-bag10' if bag else '')


def make_ctx(h, year):
    d, eng = G['d'], G['eng']
    cut = pd.Timestamp(f'{year}-01-01')
    x, _, xc = eng.static(h)
    logy = np.log(np.clip(d.y(h, 'original'), 1e-12, None))
    return K.Ctx(d.frames['original'], x, logy, eng.har_current(h, 'annual', 'original', cut), h, cut, Xc=xc)


def opt_path(h, year, scheme, loss):
    return K.OUT / 'optuna' / f'h{h}_{year}_{cid(scheme, loss, True)}.json'


def run_job(job):
    """job = (kind, h, year, scheme, [losses]); kinds: base | opt | bag | bagopt. Existing cells are kept (resume)."""
    kind, h, year, scheme, losses = job
    t0 = time.time()
    todo = [l for l in losses if not E.cell_path(h, year, cid(scheme, l, kind in ('opt', 'bagopt'), kind.startswith('bag'))).exists()]
    if not todo:
        return job, 0.
    c = make_ctx(h, year)
    for loss in todo:
        groups = tuned = None
        if kind == 'opt':
            groups = S.clusters(c.b, c.b.volume.values, c.cut)
            tuned, rows = {}, []
            for g in sorted(set(groups.values())):
                tuned[g], r = S.tune(c, scheme, loss, groups, g)
                rows += r
            p = opt_path(h, year, scheme, loss)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(dict(groups=groups, tuned=tuned, trials=rows)))
        if kind == 'bagopt':
            j = json.loads(opt_path(h, year, scheme, loss).read_text())
            groups, tuned = j['groups'], {int(k): v for k, v in j['tuned'].items()}
        seeds = SEEDS if kind.startswith('bag') else (42,)
        runs = [S.scheme_parts(c, scheme, loss, s, groups, tuned) for s in seeds]
        m = S.average([r[0] for r in runs]) if len(runs) > 1 else runs[0][0]
        f = S.finalize(c, loss, m, runs[0][1])
        E.write_cell(h, year, cid(scheme, loss, kind in ('opt', 'bagopt'), kind.startswith('bag')), c, f)
    return job, time.time() - t0


def run_pool(jobs, log, procs=8):
    K.disk_guard('jobs ' + str(len(jobs)))
    t0 = time.time()
    with mp.get_context('fork').Pool(procs) as pool:
        for i, (job, sec) in enumerate(pool.imap_unordered(run_job, jobs), 1):
            with open(log, 'a') as f:
                f.write(f'{pd.Timestamp.now(tz="UTC").isoformat()} {i}/{len(jobs)} {job[0]} h{job[1]} {job[2]} {job[3]} {job[4]} {sec:.1f}s\n')
    return time.time() - t0
