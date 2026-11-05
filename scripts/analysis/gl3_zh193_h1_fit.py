# -*- coding: utf-8 -*-
"""Zh-193 (h = 1): fits of the B items (B1..B5, B-comb, B6 legs) and of the G1/G2 fits. Stage-1 trainer, data, purge/embargo (gl3_zh190_boost_*),
HAR_i inputs and offsets (gl3_zh190_stage23_fit) are imported and not changed; new here: variant learner (monotone, linear_tree, weights, CatBoost) and the cell format.
Spec = dict(name, x in {base,b1}, loss, mono, lt, hl, cat, off in {harj,hari}, g2, scheme, seed)."""
import multiprocessing as mp
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import gl3_r56_engine as B
import gl3_zh190_boost_core as K
import gl3_zh190_boost_jobs as J
import gl3_zh190_boost_schemes as S
import gl3_zh190_stage23_core as Q
import gl3_zh190_stage23_fit as F
import gl3_zh193_h1_bars as BR

OUT = BR.OUT
TAB = BR.ROOT / 'docs/tables/gl3_zh193_h1'
EVENTS = TAB / 'fit_events.jsonl'
MONO = ('ln_rv_d', 'ln_rv_w', 'ln_rv_m')
K.HUBER_MULT.setdefault('huber8', 8.)
G = {}


def new_spec(name, **kw):
    return {**dict(name=name, x='base', loss='huber4', mono=False, lt=False, hl=None, cat=None, off='harj', g2=False, scheme='ticker', seed=42), **kw}


def ledger(h, kind, cut, n, maxend, phase='fit', detail=None):
    Q.EVENTS = EVENTS
    Q.record(h, kind, cut, n, maxend, phase, detail)


def disk_guard(stage):
    TAB.mkdir(parents=True, exist_ok=True)
    import shutil
    if shutil.disk_usage('/').free < 20 * 1024 ** 3:
        raise RuntimeError('DISK_STOP: less than 20 GiB free')


def setup():
    d, eng = K.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    B.EVENTS, B.disk_guard = EVENTS, disk_guard
    J.G.update(d=d, eng=eng)
    F.G.update(d=d, eng=eng, hari={1: F.load_hari(1)})
    G.update(d=d, eng=eng, b=d.frames['original'])
    return d


def b1_matrix():
    p = OUT / 'b1_features.parquet'
    if not p.exists():
        b = G['b']
        BR.assemble(BR.build(sorted(b.ticker.unique())), b).assign(date=b.date.values, ticker=b.ticker.values).to_parquet(p, index=False)
    f = pd.read_parquet(p)
    b = G['b']
    assert (f.date.values == b.date.values).all() and (f.ticker.values == b.ticker.values).all()
    return f[BR.NEW].values.astype(np.float32)


def hxp_columns(Y):
    """PIT HAR-X+ h=1 of the cutoff Y: in-window fit on training rows, forecast on the rows after the cut (gl3_r6b_pit, unchanged)."""
    import gl3_r6b_pit as P
    d, b = G['d'], G['b']
    H = np.nan_to_num(d.harx(1, 'original'), nan=0., posinf=0., neginf=0.)
    rec = lambda kind, cut, n, mx, phase, detail=None: ledger(1, kind, cut, n, mx, phase, detail)
    pred, _ = P.hxp_predict(H, P.Extras(b), d.y(1, 'original'), b.date.values, b._end1.values, pd.Timestamp(f'{Y}-01-01'), rec)
    return pred


def make_ctx(spec, Y):
    c = F.make_ctx(1, Y) if spec['off'] == 'hari' else J.make_ctx(1, Y)
    names = list(G['eng'].static(1)[1])
    X = c.X
    if spec['x'] == 'b1':
        X, names = np.hstack([X, G['b1']]), names + BR.NEW
    if spec['g2']:
        har, pick, inwin = F.G['hari'][1]
        lh = F.build_off(G['b'], har, inwin, pick, Y)
        X, names = np.column_stack([X, lh, hxp_columns(Y)]).astype(np.float32), names + ['ln_hari', 'ln_hxp']
        assert np.isfinite(X[c.tr][:, -2:]).all() and np.isfinite(X[c.pw][:, -2:]).all()
    c.X, c.names = X, names
    c.Xc = np.hstack([X, c.Xc[:, -2:]])  # ticker / sector codes of the cat scheme stay the last two columns
    c.w = None
    if spec['hl']:
        c.w = half_life_weights(c.b.date, c.cut, spec['hl'])
    return c


def ds(c, X, m, extra, ref=None, w=True):
    return lgb.Dataset(X[m], label=c.res[m], weight=(c.w[m] if (w and c.w is not None) else None), reference=ref, free_raw_data=False,
                       params=None if ref is not None else {**c.dsp, **extra})


def fit_lgb(c, X, gm, loss, params, seed, tag, extra=None):
    """ES on the inner split (O2) then refit on all outer-train rows with the same number of trees (as K.learn); optional weights / linear_tree."""
    extra = extra or {}
    ti, vi, tr = c.ti & gm, c.vi & gm, c.tr & gm
    dtr = ds(c, X, ti, extra)
    es = K.train_es(c, dtr, ds(c, X, vi, extra, dtr, False), loss, {**params, **extra}, seed, K.ROUNDS, K.PATIENCE)
    trees = max(1, es.best_iteration)
    c.rec(tag, seed, 'inner_ES', ti, dict(trees=trees, loss=loss))
    full = lgb.train(K.lgb_params(c, loss, {**params, **extra}, seed), ds(c, X, tr, extra), trees)
    c.rec(tag, seed, 'refit', tr, dict(trees=trees, loss=loss))
    return K.trim(es, trees), full, trees


def fit_cat(c, X, gm, depth, seed, tag):
    from catboost import CatBoostRegressor
    ti, vi, tr = c.ti & gm, c.vi & gm, c.tr & gm
    p = dict(loss_function=f'Huber:delta={K.HUBER_ALPHA * 4}', boosting_type='Ordered', depth=depth, learning_rate=.03, l2_leaf_reg=5., thread_count=1,
             random_seed=seed, verbose=False)
    es = CatBoostRegressor(iterations=K.ROUNDS, od_type='Iter', od_wait=K.PATIENCE, **p).fit(X[ti], c.res[ti], eval_set=(X[vi], c.res[vi]), use_best_model=True)
    trees = max(1, int(es.get_best_iteration()) + 1)
    c.rec(tag, seed, 'inner_ES', ti, dict(trees=trees, depth=depth))
    full = CatBoostRegressor(iterations=trees, **p).fit(X[tr], c.res[tr])
    c.rec(tag, seed, 'refit', tr, dict(trees=trees, depth=depth))
    return es, full


def mono_vec(names, top):
    """+1 on ln RV_d, ln RV_w, ln RV_m (forecast does not decrease), 0 elsewhere; indices refer to the feature subset `top`."""
    return [1 if names[j] in MONO else 0 for j in top]


def half_life_weights(dates, cut, years):
    """B4: exp(-ln2 * age / T), age in years counted from the cutoff date; every training row is older than the cut, so w <= 1."""
    age = np.asarray((pd.Timestamp(cut) - pd.DatetimeIndex(dates)).days, float) / 365.25
    return np.exp(-np.log(2) * age / years)


def ticker_parts(c, s):
    """B-b scheme (panel on the 40 best-gain features of the same cutoff -> one model per ticker) with the variant options of the spec."""
    loss, seed, tag = s['loss'], s['seed'], f'zh193_{s["name"]}'
    allm = np.ones(len(c.tk), bool)
    _, pfull, _ = fit_lgb(c, c.X, allm, loss, K.PANEL, seed, tag + '_panel')
    top = np.sort(np.argsort(-pfull.feature_importance('gain'), kind='stable')[:K.TOPK])
    Xt, parts = c.X[:, top], []
    extra = {}
    if s['mono']:
        extra['monotone_constraints'] = mono_vec(c.names, top)
    if s['lt']:
        extra['linear_tree'] = True
    for t in sorted(set(c.tk)):
        m = c.tk == t
        if s['cat']:
            es, full = fit_cat(c, Xt, m, s['cat'], seed, tag)
            gv_f, g_f = (lambda i: es.predict(Xt[i])), (lambda i: full.predict(Xt[i]))
        else:
            es, full, _ = fit_lgb(c, Xt, m, loss, {**K.TICKER_PARAMS}, seed, tag, extra)
            gv_f, g_f = (lambda i: es.predict(Xt[i], raw_score=True)), (lambda i: full.predict(Xt[i], raw_score=True))
        vi, pw = np.flatnonzero(c.vi & m), np.flatnonzero(c.pw & m)
        parts.append(dict(vi=vi, gv=gv_f(vi), pw=pw, g=g_f(pw)))
    return S.merge(parts)


def cell_path(name, Y):
    return OUT / 'cells' / f'{name}_{Y}.parquet'


def run_cell(job):
    s, Y = job
    t0 = time.time()
    if cell_path(s['name'], Y).exists():
        return job, 0.
    c = make_ctx(s, Y)
    if s['scheme'] in ('cat', 'cont', 'panel'):  # stage-1 legs reproduced by the stage-1 module (B6)
        m, unit = S.scheme_parts(c, s['scheme'], s['loss'], s['seed'])
    else:
        m, unit = ticker_parts(c, s), c.tk
    f = S.finalize(c, s['loss'], m, unit)
    pw, vi = m['pw'], m['vi']
    df = pd.DataFrame(dict(date=c.b.date.values[pw], ticker=c.tk[pw], raw=f['raw'].astype('float32'), s2=f['s2'].astype('float32'),
                           pred=f['pred'].astype('float32'), f=m['g'].astype('float32'), ln_har=c.off[pw]))
    if s['off'] == 'hari':
        val = [pd.DataFrame(dict(c=cc, ticker=x.index, s2_t=x.values, s2_pool=p)) for cc in Q.CS for x, p in [Q.val_s2(c.res[vi], m['gv'], c.tk[vi], cc)]]
        pd.concat(val).to_parquet(cell_path(s['name'] + '_val', Y), index=False)
    cell_path(s['name'], Y).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cell_path(s['name'], Y).with_suffix('.tmp'), index=False)
    cell_path(s['name'], Y).with_suffix('.tmp').replace(cell_path(s['name'], Y))
    return job, time.time() - t0


def run_pool(specs, log, procs=5):
    """procs <= 5 (conductor limit of 05.11); one process = one thread."""
    jobs = [(s, Y) for s in specs for Y in Q.ALL_YEARS]
    disk_guard('zh193 fits ' + str(len(jobs)))
    t0 = time.time()
    with mp.get_context('fork').Pool(min(procs, 5)) as pool:
        for i, (job, sec) in enumerate(pool.imap_unordered(run_cell, jobs), 1):
            with open(log, 'a') as f:
                f.write(f'{pd.Timestamp.now(tz="UTC").isoformat()} {i}/{len(jobs)} {job[0]["name"]} {job[1]} {sec:.1f}s\n')
    return time.time() - t0


def read_cells(name, years=Q.ALL_YEARS):
    return pd.concat([pd.read_parquet(cell_path(name, Y)) for Y in years], ignore_index=True)
