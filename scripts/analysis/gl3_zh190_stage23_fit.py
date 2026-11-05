# -*- coding: utf-8 -*-
"""Zh-190 stages 2-3: HAR_i inputs (LOYO lambda) and the G-1 / G-2 residual fits of the B*_h scheme (stage-1 trainer reused, not changed).
Offset = ln HAR_i without correction: in-window fitted values of the cutoff for training rows, the cutoff's own forecast for the prediction year."""
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

ET = K.OUT / 'etalony'
ETT = Q.ROOT / 'docs/tables/gl3_zh190/etalony'
G = {}  # data, engine, HAR_i inputs: loaded before the fork


def lam_tag(lam):
    return f'lam{int(round(lam * 100))}'


def load_hari(h):
    """-> (forecast frame date, ticker, level [with s2*/2], ln_raw [without]; LOYO lambda by year (2016 takes 2017); in-window frame)."""
    pick = pd.read_csv(ETT / f'loyo_pick_h{h}.csv').set_index('year').lam.to_dict()
    pick[2016] = pick[2017]
    s2, parts = pd.read_csv(ETT / f'har_i_s2_h{h}.csv'), []
    for Y, lam in pick.items():
        f = pd.read_parquet(ET / f'HARi_{lam_tag(lam)}_h{h}.parquet')
        f = f[f.date.dt.year == Y].copy()
        s = s2[s2.cutoff == Y].set_index('ticker')
        star = lam * s.s2_i + (1 - lam) * s.s2_pool
        f['level'] = f.prediction.astype('float64')
        f['ln_raw'] = np.log(f.level) - .5 * f.ticker.map(star)
        parts.append(f[['date', 'ticker', 'level', 'ln_raw']])
    return pd.concat(parts, ignore_index=True), pick, pd.read_parquet(ET / f'HARi_inwin_h{h}.parquet')


def build_off(b, har, inwin, pick, Y):
    """Offset ln HAR_i on every row of b: training rows (date < cut) from the in-window fit of the cutoff, prediction-year rows from the forecast."""
    key, off = pd.MultiIndex.from_frame(b[['date', 'ticker']]), np.full(len(b), np.nan)
    w = inwin[inwin.cutoff == Y]
    ix = key.get_indexer(pd.MultiIndex.from_frame(w[['date', 'ticker']]))
    off[ix[ix >= 0]] = w[f'ln_{lam_tag(pick[Y])}'].values[ix >= 0]
    p = har[har.date.dt.year == Y]
    ix = key.get_indexer(pd.MultiIndex.from_frame(p[['date', 'ticker']]))
    assert (ix >= 0).all()
    off[ix] = p.ln_raw.values
    return off


def make_ctx(h, Y):
    d, eng = G['d'], G['eng']
    cut = pd.Timestamp(f'{Y}-01-01')
    x, _, xc = eng.static(h)
    b = d.frames['original']
    har, pick, inwin = G['hari'][h]
    c = K.Ctx(b, x, np.log(np.clip(d.y(h, 'original'), 1e-12, None)), build_off(b, har, inwin, pick, Y), h, cut, Xc=xc)
    assert np.isfinite(c.off[c.tr]).all() and np.isfinite(c.off[c.pw]).all(), 'offset not defined on all training / prediction rows'
    return c


def cell_path(h, Y, kind, part=''):
    return Q.OUT / 'cells' / f'h{h}_{Y}_{kind}{part}.parquet'


def run_fit(job):
    """job = (kind, h, year); kind g1 = loss of B*_h on the residual, g2 = QLIKE on the residual. Existing cells are kept."""
    kind, h, Y = job
    t0 = time.time()
    if cell_path(h, Y, kind).exists():
        return job, 0.
    scheme, loss, bag, _ = Q.parse_config(G['bstar'][h])
    loss = 'qlike' if kind == 'g2' else loss
    c = make_ctx(h, Y)
    runs = [S.scheme_parts(c, scheme, loss, s) for s in (J.SEEDS if bag else (42,))]
    m = S.average([r[0] for r in runs]) if bag else runs[0][0]
    pw, vi = m['pw'], m['vi']
    df = pd.DataFrame(dict(date=c.b.date.values[pw], ticker=c.tk[pw], ln_har=c.off[pw], f=m['g'].astype('float32')))
    val = []
    for cc in Q.CS:
        s2t, pool = Q.val_s2(c.res[vi], m['gv'], c.tk[vi], cc)
        val.append(pd.DataFrame(dict(c=cc, ticker=s2t.index, s2_t=s2t.values, s2_pool=pool)))
    for path, frame in ((cell_path(h, Y, kind), df), (cell_path(h, Y, kind, '_val'), pd.concat(val))):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path.with_suffix('.tmp'), index=False)
        path.with_suffix('.tmp').replace(path)
    return job, time.time() - t0


def disk_guard(stage):
    Q.TAB.mkdir(parents=True, exist_ok=True)
    out = subprocess.run(['df', '-h', '/'], capture_output=True, text=True, check=True).stdout
    with (Q.TAB / 'disk.log').open('a') as f:
        f.write(str(pd.Timestamp.now(tz='UTC')) + ' ' + stage + '\n' + out)
    if shutil.disk_usage('/').free < 20 * 1024 ** 3:
        raise RuntimeError('DISK_STOP: less than 20 GiB free')


def setup():
    """Stage-1 setup, but the ledger and the disk log go to stage23 (stage 1 files are not written)."""
    d, eng = K.setup()
    Q.TAB.mkdir(parents=True, exist_ok=True)
    Q.OUT.mkdir(parents=True, exist_ok=True)
    B.EVENTS, B.disk_guard = Q.EVENTS, disk_guard
    G.update(d=d, eng=eng, bstar=pd.read_csv(K.TAB / 'bstar.csv').set_index('h').config.to_dict(), hari={h: load_hari(h) for h in Q.HS})
    return d


def jobs_for(hs=Q.HS):
    out = []
    for h in hs:
        kinds = ('g1',) if Q.parse_config(G['bstar'][h])[1] == 'qlike' else ('g1', 'g2')  # QLIKE B*: G-1 and G-2 share the fit
        out += [(k, h, Y) for Y in Q.ALL_YEARS for k in kinds]
    return sorted(out, key=lambda j: (-(Q.parse_config(G['bstar'][j[1]])[2]), j[1] != 1, j[1], j[2]))


def run_pool(jobs, log, procs=8):
    disk_guard(f'fit jobs {len(jobs)}')
    t0 = time.time()
    with mp.get_context('fork').Pool(procs) as pool:
        for i, (job, sec) in enumerate(pool.imap_unordered(run_fit, jobs), 1):
            with open(log, 'a') as f:
                f.write(f'{pd.Timestamp.now(tz="UTC").isoformat()} {i}/{len(jobs)} {job} {sec:.1f}s\n')
    return time.time() - t0
