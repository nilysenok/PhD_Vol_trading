# -*- coding: utf-8 -*-
"""Zh-190 stage 1, core: one LightGBM trainer on the residual ln RV(t+h) - HAR-J offset.

Reuses K9/K10 (gl3_r56_engine: offset by Engine.har_current, B.qlike_grad), purge/embargo (gl3_r6h5_engine.gap_boundary),
data and masks (gl3_r56_data). New here: the Dataset cache, the ticker and continuation (init_model) schemes (schemes module).
"""
import json
import shutil
import subprocess
import numpy as np
import pandas as pd
import lightgbm as lgb
import gl3_r56_engine as B
import gl3_r6h5_engine as R6
import gl3_r56_data as D0
from gl3_r56_data import Data, SECTORS, ROOT
from pathlib import Path
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'

_read_base = D0.read
B.record = R6._record_base  # the h5 wrapper tags records with the round-6 contract; not wanted here
TAB = ROOT / 'docs/tables/gl3_zh190/stage1'
OUT = ROOT / 'data/regen/gl3_zh190'
HS = (1, 5, 22)
SCHEMES = ('panel', 'ticker', 'cat', 'cont')
LOSSES = ('mse', 'huber1', 'huber2', 'huber4', 'qlike')
# P4: the final's LightGBM huber has no explicit alpha (l2_lib.py:651 objective='huber'), so LightGBM's default 0.9 is the threshold.
HUBER_ALPHA = 0.9
HUBER_MULT = {'huber1': 1., 'huber2': 2., 'huber4': 4.}
ROUNDS, PATIENCE, TOPK, VAL_DATES = 1000, 50, 40, 126
PANEL = dict(num_leaves=31, learning_rate=.03, min_data_in_leaf=100, lambda_l2=10., feature_fraction=.8, bagging_fraction=.8)
TICKER_PARAMS = dict(max_depth=4, num_leaves=15, min_data_in_leaf=50, lambda_l2=5., learning_rate=.03, feature_fraction=.8, bagging_fraction=.8)
CONT_PARAMS = {**TICKER_PARAMS, 'learning_rate': .01}
CONT_ROUNDS, CONT_PATIENCE, CONT_MIN = 200, 20, 50


def disk_guard(stage):
    TAB.mkdir(parents=True, exist_ok=True)
    out = subprocess.run(['df', '-h', '/'], capture_output=True, text=True, check=True).stdout
    with (TAB / 'disk.log').open('a') as f:
        f.write(str(pd.Timestamp.now(tz='UTC')) + ' ' + stage + '\n' + out)
    if shutil.disk_usage('/').free < 20 * 1024 ** 3:
        raise RuntimeError('DISK_STOP: less than 20 GiB free')


def panel_params(fin):
    """fin5params['lgbm'] -> LightGBM parameters (same mapping as Engine.boost warm start)."""
    m = [('num_leaves', 'num_leaves', int), ('learning_rate', 'learning_rate', float), ('min_data_in_leaf', 'min_data_in_leaf', int),
         ('bagging_fraction', 'bagging_fraction', float), ('feature_fraction', 'feature_fraction', float),
         ('lambda_l1', 'reg_alpha', float), ('lambda_l2', 'reg_lambda', float)]
    return {d: f(fin[s]) for d, s, f in m}


class Eng(B.Engine):
    """Engine without the cache directory / budget side effects; only har_current and the data are used."""
    def __init__(self, data):
        self.data = data
        self.har, self.har_oos, self.har_calib, self.hxp, self.pc, self.rawcache, self.predcache = {}, {}, {}, {}, {}, {}, {}
        self.deadline = 1e18
        self.xs = {}

    def static(self, h):
        if h not in self.xs:
            x, names = self.data.static_x(h, 'original', False, False)
            b = self.data.frames['original']
            tcode = b.ticker.map({t: i for i, t in enumerate(sorted(b.ticker.unique()))}).values
            scode = b.ticker.map(SECTORS).map({s: i for i, s in enumerate(sorted(set(SECTORS.values())))}).values
            self.xs[h] = (x, names, np.column_stack([x, tcode, scode]).astype(np.float32))
        return self.xs[h]


def _read_without_alt(path, cols, filters=None):
    """The 'alt' measure of rounds 5-6 (grid10_shift_mean.parquet) is not in the repository and is not used here: empty stand-in."""
    if Path(path).name == 'grid10_shift_mean.parquet':
        return pd.DataFrame({c: pd.Series(dtype='datetime64[ns]' if c == 'date' else 'object' if c == 'ticker' else float) for c in cols})
    return _read_base(path, cols, filters)


def setup():
    global PANEL
    TAB.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    B.disk_guard = disk_guard
    B.EVENTS = TAB / 'fit_events.jsonl'
    D0.read = _read_without_alt
    d = Data()
    PANEL = panel_params(d.fin5params['lgbm'])
    return d, Eng(d)


def purged_h(b, h, cut):
    """Training rows of a refit at cut: end of target before the cut shifted back h dates (purge + embargo h), as purged() of 6/6b."""
    edge = R6.gap_boundary(b, cut, h)
    y = b[f'_y{h}'].values
    return ((b.date < cut) & (b[f'_end{h}'] < edge) & np.isfinite(y) & (y > 1e-12)).values


class _Shim:
    """Dataset view for B.qlike_grad in residual coordinates: zero offset, no weights."""
    def __init__(self, ds): self.ds = ds
    def get_init_score(self):
        s = self.ds.get_init_score()
        return 0. if s is None else s
    def get_label(self): return self.ds.get_label()
    def get_weight(self): return None


def qlike_obj(pred, ds):
    return B.qlike_grad(pred, _Shim(ds), 1.)


def qlike_eval(pred, ds):
    r = np.exp(np.clip(ds.get_label() - pred, -60, 60))
    return 'QLIKE', float(np.mean(r - np.log(r) - 1)), False


class Ctx:
    """One (h, cutoff): residual target, row masks (outer purge, inner ES split) and the Dataset cache."""
    def __init__(self, b, X, logy, off, h, cut, Xc=None, compat=False, tr=None):
        self.b, self.h, self.cut, self.X, self.Xc, self.off = b, h, pd.Timestamp(cut), X, Xc, off
        self.res, self.logy = logy - off, logy
        self.tk = b.ticker.values
        self.threads, self.max_bin, self.compat = (2, 255, True) if compat else (1, 63, False)
        self.cache = {}
        # feature_pre_filter=False lets Optuna vary min_data_in_leaf on one Dataset; compat keeps the K9 default (it changes feature_fraction sampling)
        self.dsp = dict(max_bin=self.max_bin, feature_pre_filter=compat, verbosity=-1)
        end =b[f'_end{h}']
        self.tr = purged_h(b, h, self.cut) if tr is None else tr
        u = np.unique(b.date.values[self.tr])
        self.edge = pd.Timestamp(u[max(1, len(u) - VAL_DATES)])
        self.vi = self.tr & (b.date >= self.edge).values
        lim = self.edge if compat else R6.gap_boundary(b, self.edge, h)  # compat = K9 split (no embargo)
        self.ti = self.tr & (b.date < self.edge).values & (end < lim).values
        self.pw = ((b.date >= self.cut) & (b.date < self.cut + pd.DateOffset(years=1))).values
        assert self.ti.sum() > 50 and self.vi.sum() > 20

    def rec(self, kind, seed, phase, mask, detail=None):
        B.record(kind, self.h, self.cut, 'annual', 'original', seed, int(mask.sum()), self.b.loc[mask, f'_end{self.h}'].max(), phase=phase, detail=detail)


def lgb_params(c, loss, params, seed):
    p = dict(bagging_freq=1, num_threads=c.threads, verbosity=-1, seed=seed, force_col_wise=True, max_bin=c.max_bin, **params)
    if loss == 'qlike':
        p.update(objective=qlike_obj, metric='None')
    elif loss == 'mse':
        p.update(objective='regression', metric='l2')
    else:
        p.update(objective='huber', metric='huber', alpha=HUBER_ALPHA * HUBER_MULT[loss])
    return p


def train_es(c, dtr, dva, loss, params, seed, rounds, patience, init=None):
    kw = {'feval': qlike_eval} if loss == 'qlike' else {}
    return lgb.train(lgb_params(c, loss, params, seed), dtr, rounds, valid_sets=[dva], init_model=init,
                     callbacks=[lgb.early_stopping(patience, verbose=False)], **kw)


def trim(booster, trees):
    """Exactly `trees` iterations, no best_iteration state: a clean init_model / predictor."""
    return lgb.Booster(model_str=booster.model_to_string(num_iteration=trees))


def k10_form(c, loss):
    """Only the G2 check builds QLIKE exactly as K10 does (label ln RV, init_score = offset); production uses the residual form, which is
    the same loss but lets one Dataset serve all losses and init_model."""
    return c.compat and loss == 'qlike'


def dataset(c, X, mask, cats, loss, ref=None):
    q = k10_form(c, loss)
    return lgb.Dataset(X[mask], label=(c.logy if q else c.res)[mask], init_score=c.off[mask] if q else None, reference=ref,
                       free_raw_data=False, categorical_feature=list(cats) or 'auto', params=c.dsp if ref is None else None)


def learn(c, X, loss, params, gm, key, seed=42, cats=(), rounds=ROUNDS, patience=PATIENCE, refit=True, tag=''):
    """ES on the inner split (O2), then refit on all outer-train rows with the same number of trees. Datasets cached by key."""
    ti, vi, tr = c.ti & gm, c.vi & gm, c.tr & gm
    k = (key, tuple(cats), k10_form(c, loss))
    if k not in c.cache:
        dtr = dataset(c, X, ti, cats, loss)
        c.cache[k] = (dtr, dataset(c, X, vi, cats, loss, dtr))
    dtr, dva = c.cache[k]
    es = train_es(c, dtr, dva, loss, params, seed, rounds, patience)
    trees = max(1, es.best_iteration)
    c.rec(tag, seed, 'inner_ES', ti, dict(trees=trees, loss=loss))
    out = dict(es=trim(es, trees), trees=trees, full=None)
    if refit:
        dfull = dataset(c, X, tr, cats, loss)
        out['full'] = lgb.train(lgb_params(c, loss, params, seed), dfull, trees)
        c.rec(tag, seed, 'refit', tr, dict(trees=trees, loss=loss))
    return out
