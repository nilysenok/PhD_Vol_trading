# -*- coding: utf-8 -*-
"""Zh-190 stage 1: schemes panel (B-a), ticker (B-b), cat (B-v), cont (B-g); O7 correction; Optuna per ticker group (P5)."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.cluster import KMeans
import gl3_r56_engine as B
import gl3_r6h5_engine as R6
import gl3_zh190_boost_core as K

optuna.logging.set_verbosity(optuna.logging.WARNING)


def panel_L(c, loss, params, gid, gm, seed, cat=False, refit=True):
    """Cached panel learner (ES model, refit model, trees) of one group."""
    k = ('L', loss, gid, seed, cat, json.dumps(params, sort_keys=True), refit)
    if k not in c.cache:
        X = c.Xc if cat else c.X
        cats = (X.shape[1] - 2, X.shape[1] - 1) if cat else ()
        c.cache[k] = K.learn(c, X, loss, params, gm, ('P', gid, cat), seed, cats, refit=refit, tag=f'zh190_{"cat" if cat else "panel"}_{loss}')
    return c.cache[k]


def _part(X, L, c, gm, refit):
    vi = np.flatnonzero(c.vi & gm)
    part = dict(vi=vi, gv=L['es'].predict(X[vi], raw_score=True))
    if refit:
        pw = np.flatnonzero(c.pw & gm)
        part.update(pw=pw, g=L['full'].predict(X[pw], raw_score=True))
    return part


def part_panel(c, loss, params, gid, gm, seed, cat, refit):
    L = panel_L(c, loss, params, gid, gm, seed, cat, refit)
    return [_part(c.Xc if cat else c.X, L, c, gm, refit)]


def part_ticker(c, loss, params, gid, gm, seed, refit):
    """B-b: one model per ticker on the 40 best-gain features of the same-cutoff panel (default panel params)."""
    L = panel_L(c, loss, K.PANEL, gid, gm, seed, False, True)
    top = np.sort(np.argsort(-L['full'].feature_importance('gain'), kind='stable')[:K.TOPK])
    Xt, p, parts = c.X[:, top], {**K.TICKER_PARAMS, **params}, []
    for t in sorted(set(c.tk[gm])):
        m = gm & (c.tk == t)
        Lt = K.learn(c, Xt, loss, p, m, ('T', t, top.tobytes()), seed, (), refit=refit, tag=f'zh190_ticker_{loss}')
        parts.append(_part(Xt, Lt, c, m, refit))
    return parts


def part_cont(c, loss, params, gid, gm, seed, refit):
    """B-g: panel (tuned params) -> continuation per ticker with init_model, lr .01, <=200 trees, ES 20, >=50 trees."""
    L = panel_L(c, loss, {**K.PANEL, **params}, gid, gm, seed, False, refit)
    X, es0, parts = c.X, L['es'], []
    n0 = es0.current_iteration()
    for t in sorted(set(c.tk[gm])):
        m = gm & (c.tk == t)
        ti, vi, tr = c.ti & m, c.vi & m, c.tr & m
        dtr = lgb.Dataset(X[ti], label=c.res[ti], free_raw_data=False, params=c.dsp)
        dva = lgb.Dataset(X[vi], label=c.res[vi], reference=dtr, free_raw_data=False)
        b1 = K.train_es(c, dtr, dva, loss, K.CONT_PARAMS, seed, K.CONT_ROUNDS, K.CONT_PATIENCE, init=es0)
        extra = int(np.clip(b1.best_iteration - n0, K.CONT_MIN, K.CONT_ROUNDS))
        c.rec(f'zh190_cont_{loss}', seed, 'inner_ES', ti, dict(extra=extra, panel_trees=n0, ticker=t))
        part = dict(vi=np.flatnonzero(vi), gv=b1.predict(X[vi], raw_score=True, num_iteration=n0 + extra))
        if refit:
            dfull = lgb.Dataset(X[tr], label=c.res[tr], free_raw_data=False, params=c.dsp)
            bf = lgb.train(K.lgb_params(c, loss, K.CONT_PARAMS, seed), dfull, extra, init_model=L['full'])
            c.rec(f'zh190_cont_{loss}', seed, 'refit', tr, dict(extra=extra, panel_trees=n0, ticker=t, total=bf.current_iteration()))
            pw = np.flatnonzero(c.pw & m)
            part.update(pw=pw, g=bf.predict(X[pw], raw_score=True))
        parts.append(part)
    return parts


def scheme_parts(c, scheme, loss, seed=42, groups=None, tuned=None, only=None, refit=True):
    """Parts of (scheme, loss); groups: ticker -> group id (None = one group); tuned: group id -> LightGBM params of the main learner."""
    gid = np.array([groups[t] for t in c.tk]) if groups else np.zeros(len(c.tk), int)
    parts = []
    for g in (sorted(set(gid)) if only is None else [only]):
        gm, prm = gid == g, (tuned or {}).get(g, {})
        if scheme in ('panel', 'cat'):
            parts += part_panel(c, loss, {**K.PANEL, **prm}, g, gm, seed, scheme == 'cat', refit)
        elif scheme == 'ticker':
            parts += part_ticker(c, loss, prm, g, gm, seed, refit)
        else:
            parts += part_cont(c, loss, prm, g, gm, seed, refit)
    return merge(parts), (c.tk if scheme in ('ticker', 'cont') else gid)


def merge(parts):
    out = {}
    for key, val in (('vi', 'gv'), ('pw', 'g')):
        if key in parts[0]:
            i = np.concatenate([p[key] for p in parts])
            o = np.argsort(i, kind='stable')
            out[key], out[val] = i[o], np.concatenate([p[val] for p in parts])[o]
    return out


def average(ms):
    """Seed bagging: mean of the log forecasts (P6)."""
    out = {k: ms[0][k] for k in ms[0] if k in ('vi', 'pw')}
    for k in ('gv', 'g'):
        if k in ms[0]:
            out[k] = np.mean([m[k] for m in ms], axis=0)
    return out


def correction(c, m, unit):
    """O7: s2 = variance (ddof 0) of the log error on the inner validation, one scalar per unit (group / ticker)."""
    err = c.res[m['vi']] - m['gv']
    return pd.Series(err).groupby(unit[m['vi']]).apply(lambda e: float(np.var(e.values)))


def finalize(c, loss, m, unit):
    """Level forecast exp(raw + s2/2) for the prediction window; QLIKE-in-logs gets no correction (P3)."""
    s2 = correction(c, m, unit)
    out = dict(s2_unit=s2)
    if 'pw' in m:
        s = s2.reindex(unit[m['pw']]).values
        raw = c.off[m['pw']] + m['g']
        corr = np.zeros(len(s)) if loss == 'qlike' else s / 2
        out.update(pw=m['pw'], raw=raw, s2=s, corr=corr, pred=B.level(raw + corr))
    return out


def val_qlike(c, loss, m, unit):
    """QLIKE of the level forecast on the inner validation (Optuna objective, P5)."""
    s2 = correction(c, m, unit).reindex(unit[m['vi']]).values
    r = np.exp(np.clip(c.res[m['vi']] - m['gv'] - (0. if loss == 'qlike' else s2 / 2), -60, 60))
    return float(np.mean(r - np.log(r) - 1))


def suggest(trial):
    p = {}
    for name, lo, hi, kind in R6.SPACE['lgb']:
        dest = {'reg_alpha': 'lambda_l1', 'reg_lambda': 'lambda_l2'}.get(name, name)
        p[dest] = trial.suggest_int(name, lo, hi, log=kind == 'logint') if kind in ('int', 'logint') else trial.suggest_float(name, lo, hi, log=kind == 'log')
    return p


def tune(c, scheme, loss, groups, gid, n_trials=20, seed=42):
    """TPE (seed 42), 20 trials per group; returns best params and the trial rows."""
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    rows = []
    for _ in range(n_trials):
        t = study.ask()
        prm = suggest(t)
        m, unit = scheme_parts(c, scheme, loss, seed, groups, {gid: prm}, only=gid, refit=False)
        q = val_qlike(c, loss, m, unit)
        study.tell(t, q)
        rows.append(dict(cutoff=str(c.cut.date()), scheme=scheme, loss=loss, group=int(gid), trial=t.number, qlike_val=q, params=prm))
    return min(rows, key=lambda r: r['qlike_val'])['params'], rows


def clusters(b, vol, cut, k=3):
    """P5: k-means (seed 42, n_init 20) on standardised median ln RV and median ln turnover over 252 dates before the cut."""
    f = []
    full = b.assign(ln_turn=np.log(np.clip(b.close.values * np.asarray(vol), 1e-12, None)), ln_rv=np.log(np.clip(b.rv.values, 1e-12, None)))
    past = full[full.date < cut]
    for t, g in past.groupby('ticker'):
        g = g.tail(252)
        f.append((t, g.ln_rv.median(), g.ln_turn.median()))
    d = pd.DataFrame(f, columns=['ticker', 'ln_rv', 'ln_turn']).set_index('ticker')
    z = (d - d.mean()) / d.std(ddof=0)
    lab = KMeans(k, random_state=42, n_init=20).fit_predict(z.values)
    rank = pd.Series(d.ln_rv.values).groupby(lab).mean().rank().astype(int) - 1
    return {t: int(rank[l]) for t, l in zip(d.index, lab)}
