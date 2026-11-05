# -*- coding: utf-8 -*-
"""Zh-194 (d): stochastic links of the chain retrained with seed 42-46: B*_h5 (cat-qlike, stage-1 trainer) and the G-1 residual fits of B*_h
(h=5 cat-qlike, h=22 cat-huber1-bag10). Stage-1 / stage-2 trainers are reused unchanged; only the seed is a parameter.
Seed s: h=5 -> LightGBM seed s; h=22 -> the 10 bag seeds 42..51 of the stage-1 bag shifted by 1000*(s-42) (s = 42 reproduces the original)."""
import multiprocessing as mp
import shutil
import subprocess
import time
import numpy as np
import pandas as pd
import gl3_r56_engine as B
import gl3_zh190_boost_core as K
import gl3_zh190_boost_jobs as J
import gl3_zh190_boost_schemes as S
import gl3_zh190_stage23_core as Q
import gl3_zh190_stage23_fit as F
import gl3_zh194_core as Z


def bag_seeds(s):
    return tuple(x + 1000 * (s - 42) for x in J.SEEDS)


def cell(kind, h, Y, s, part=''):
    return Z.OUT / 'cells' / f'h{h}_{Y}_{kind}_s{s}{part}.parquet'


def disk_guard(stage):
    Z.TAB.mkdir(parents=True, exist_ok=True)
    out = subprocess.run(['df', '-h', '/'], capture_output=True, text=True, check=True).stdout
    with (Z.TAB / 'disk.log').open('a') as f:
        f.write(str(pd.Timestamp.now(tz='UTC')) + ' ' + stage + '\n' + out)
    if shutil.disk_usage('/').free < 20 * 1024 ** 3:
        raise RuntimeError('DISK_STOP: less than 20 GiB free')


def setup():
    """Stage-1 setup, but the ledger, the disk log and the cells go to zh194 (no stage-1 / stage-2 file is written)."""
    d, eng = K.setup()
    Z.TAB.mkdir(parents=True, exist_ok=True)
    Z.OUT.mkdir(parents=True, exist_ok=True)
    B.EVENTS, B.disk_guard = Z.EVENTS, disk_guard
    J.G.update(d=d, eng=eng)
    F.G.update(d=d, eng=eng, bstar=pd.read_csv(K.TAB / 'bstar.csv').set_index('h').config.to_dict(), hari={h: F.load_hari(h) for h in Z.HS})
    return d


def _save(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix('.tmp'), index=False)
    path.with_suffix('.tmp').replace(path)


def run_b(h, Y, s):
    """B*_h5 of the cutoff Y with seed s (QLIKE in logs: no O7 correction)."""
    scheme, loss, bag, _ = Q.parse_config(F.G['bstar'][h])
    assert h == 5 and not bag
    c = J.make_ctx(h, Y)
    m, unit = S.scheme_parts(c, scheme, loss, s)
    f = S.finalize(c, loss, m, unit)
    _save(cell('b', h, Y, s), pd.DataFrame(dict(date=c.b.date.values[f['pw']], ticker=c.tk[f['pw']], pred=f['pred'].astype('float32'))))


def run_g(h, Y, s):
    """G-1 residual fit (loss of B*_h on ln RV - ln HAR_i); for QLIKE B* it is also the base of G-2 (as in stage 2)."""
    scheme, loss, bag, _ = Q.parse_config(F.G['bstar'][h])
    c = F.make_ctx(h, Y)
    runs = [S.scheme_parts(c, scheme, loss, x) for x in (bag_seeds(s) if bag else (s,))]
    m = S.average([r[0] for r in runs]) if bag else runs[0][0]
    pw, vi = m['pw'], m['vi']
    df = pd.DataFrame(dict(date=c.b.date.values[pw], ticker=c.tk[pw], ln_har=c.off[pw], f=m['g'].astype('float32')))
    val = []
    for cc in Q.CS:
        s2t, pool = Q.val_s2(c.res[vi], m['gv'], c.tk[vi], cc)
        val.append(pd.DataFrame(dict(c=cc, ticker=s2t.index, s2_t=s2t.values, s2_pool=pool)))
    _save(cell('g', h, Y, s), df)
    _save(cell('g', h, Y, s, '_val'), pd.concat(val))


def run_job(job):
    kind, h, Y, s = job
    t0 = time.time()
    if cell(kind, h, Y, s).exists():
        return job, 0.
    (run_b if kind == 'b' else run_g)(h, Y, s)
    return job, time.time() - t0


def jobs():
    out = [(k, h, Y, s) for s in Z.SEEDS for h in Z.HS for Y in Q.ALL_YEARS for k in ('b', 'g') if not (k == 'b' and h == 22)]
    return sorted(out, key=lambda j: (-j[1], j[3], j[2]))   # h=22 (10 bags) first


def run_pool(procs=3):
    """At most 3 processes x 1 thread (conductor's limit)."""
    disk_guard('zh194 fit')
    js, t0 = jobs(), time.time()
    with mp.get_context('fork').Pool(procs) as pool:
        for i, (job, sec) in enumerate(pool.imap_unordered(run_job, js), 1):
            with open(Z.TAB / 'run.log', 'a') as f:
                f.write(f'{pd.Timestamp.now(tz="UTC").isoformat()} {i}/{len(js)} {job} {sec:.1f}s\n')
    return time.time() - t0


def read_comp(h, s, key, S23):
    """Seed-s components on the rows of `key`: B* (h=5), G-1 / G-2 points."""
    cells = pd.concat([pd.read_parquet(cell('g', h, Y, s)) for Y in Q.ALL_YEARS], ignore_index=True)
    val = pd.concat([pd.read_parquet(cell('g', h, Y, s, '_val')).assign(year=Y) for Y in Q.ALL_YEARS], ignore_index=True)
    comp = dict(g1=S23.g1_points(cells, val, key), g2=S23.g2_points(cells, key) if h == 5 else {})
    if h == 5:
        b = pd.concat([pd.read_parquet(cell('b', h, Y, s)) for Y in Q.ALL_YEARS], ignore_index=True)
        comp['bst'] = S23.on_a(key, b, 'pred')
    return comp
