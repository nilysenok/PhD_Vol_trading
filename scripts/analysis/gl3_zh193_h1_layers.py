# -*- coding: utf-8 -*-
"""Zh-193 G-items on top of B** (h = 1): G3 (per-ticker NNLS ridge over {FIN_PIT, B**, HAR_i, G1}), G4 (calm/stress weights: hard and logistic),
G5 (EWMA recalibration, half-lives 10/20/60), G6 (conditional bias exp(v/2)), chain over all G items and the seed-fixed replay for G7.
Frame `a`: rows sorted by (ticker, date), 2016-2021. Every fit at the 1 January cutoff uses rows whose target ends before the cut."""
import numpy as np
import pandas as pd
import gl3_r6b_layers as L
import gl3_zh190_stage23_core as Q
import gl3_zh190_stage23_layers as Y3

H = 1
HLS, WS = (10, 20, 60), (.5, .75)
MIN_REG = 100


def _lg(v):
    return np.log(np.clip(v, 1e-12, None))


def _regime_fit(lns, ly, tk, tr, st):
    """Weights per (ticker, regime): NNLS ridge, 0.5 own + 0.5 pool of the regime (pool of all rows if the regime is thin)."""
    coef, s2 = {}, {}
    wa, ba = Y3.nnls_ridge(lns[tr], ly[tr])
    for r in (False, True):
        mr = tr & (st == r)
        wp, bp = Y3.nnls_ridge(lns[mr], ly[mr]) if mr.sum() >= MIN_REG else (wa, ba)
        res = np.full(len(ly), np.nan)
        for t in np.unique(tk):
            m = mr & (tk == t)
            wi, bi = Y3.nnls_ridge(lns[m], ly[m]) if m.sum() >= Q.MIN_OBS else (wp, bp)
            coef[(t, r)] = (Q.SHRINK * wi + (1 - Q.SHRINK) * wp, Q.SHRINK * bi + (1 - Q.SHRINK) * bp)
            res[m] = ly[m] - (coef[(t, r)][1] + lns[m] @ coef[(t, r)][0])
        pool = float(np.var(res[mr])) if mr.sum() > 1 else 0.
        for t in np.unique(tk):
            m = mr & (tk == t)
            s2[(t, r)] = float(np.var(res[m])) if m.sum() >= Q.MIN_OBS else pool
    return coef, s2


def _regime_predict(coef, s2, lns, tk, st, mask):
    out = np.full(len(tk), np.nan)
    for (t, r), (w, b0) in coef.items():
        sel = mask & (tk == t) & (st == r)
        out[sel] = Q.level(b0 + lns[sel] @ w + s2[(t, r)] / 2)
    return out


def g4_series(a, lns, y, end, stress, base, rec=None):
    """-> (hard, calm_all, stress_all): hard = weights of the regime of day d; calm_all / stress_all = every row predicted with one regime's weights
    (inputs of the logistic mixture). Years 2016 (no history) and rows without a fit = base."""
    ly, tk, year = _lg(y), a.ticker.values, a.year.values
    hard, calm, strs = (np.asarray(base, float).copy() for _ in range(3))
    with np.errstate(all='ignore'):
        for Y in Q.YEARS:
            cut = pd.Timestamp(f'{Y}-01-01')
            tr = Q.past_mask(end, cut) & np.isfinite(ly) & (y > 1e-12) & np.isfinite(lns).all(axis=1)
            coef, s2 = _regime_fit(lns, ly, tk, tr, stress)
            va = year == Y
            hard[va] = _regime_predict(coef, s2, lns, tk, stress, va)[va]
            for flag, dest in ((False, calm), (True, strs)):
                dest[va] = _regime_predict(coef, s2, lns, tk, np.full(len(tk), flag), va)[va]  # out-of-sample rows of the cutoff's own year only
            if rec:
                rec(H, 'G4_regime', cut, int(tr.sum()), np.asarray(end)[tr].max())
    return hard, calm, strs


def g6_series(a, base, y, end, z3, rec=None):
    """exp(v/2) on the log forecast without its own bias (pooled s2/2 of the same past rows removed); v = max(0, regression of the squared
    log error on {1, ln RVI_d, ln RQ_d, ln RV_w}) over the three past closed years (as Engine.calibrate 'conditional')."""
    out = np.asarray(base, float).copy()
    ly, year, date = _lg(y), a.year.values, a.date.values
    for Y in Q.YEARS:
        cut = pd.Timestamp(f'{Y}-01-01')
        tr = Q.past_mask(end, cut) & (date >= np.datetime64(cut - pd.DateOffset(years=3))) & np.isfinite(ly) & (y > 1e-12) & np.isfinite(base)
        if tr.sum() < 50:
            continue
        lb = np.log(base)
        s2 = float(np.var(ly[tr] - lb[tr]))
        err = ly[tr] - (lb[tr] - s2 / 2)
        zz = np.column_stack([np.ones(len(a)), z3])
        good = np.isfinite(zz[tr]).all(axis=1)
        if good.sum() < 50:
            continue
        coef = np.linalg.lstsq(zz[tr][good], ((err - err.mean()) ** 2)[good], rcond=None)[0]
        va = year == Y
        with np.errstate(all='ignore'):  # spurious BLAS (Accelerate) warnings on finite data, as in 6b
            v = np.maximum(0., np.nan_to_num(zz[va], nan=0.) @ coef)
        out[va] = np.exp(lb[va] - s2 / 2 + v / 2)
        if rec:
            rec(H, 'G6_cond', cut, int(good.sum()), np.asarray(end)[tr][good].max(), detail=dict(coef=coef.tolist()))
    return out


def g5_points(a, base, y):
    return {f'hl{hl}_w{int(w * 100)}': L.layer14(a, base, y, hl, w, lag=H) for hl in HLS for w in WS}


class Ev:
    """Collects series; with a scorer (`score(p) -> dict`) it also picks LOYO points and best-of names, otherwise replays the fixed selection."""
    def __init__(self, year, score=None, sel=None, adder=None):
        self.year, self.score, self.fixed, self.sel = year, score, sel is not None, dict(sel or {})
        self.fc, self.sc, self.adder = {}, {}, adder

    def add(self, name, p, eligible=True, note=''):
        p = Q.f32(p)
        assert np.isfinite(p).all() and (p > 0).all(), name
        self.fc[name] = p
        if self.score:
            self.sc[name] = self.score(p)
        if self.adder:
            self.adder(name, p, eligible, note)
        return p

    def choose(self, tag, pts):
        for k, v in pts.items():
            self.add(f'{tag}_{k}', v, False)
        pk = ({int(k): v for k, v in self.sel[f'{tag}_pick'].items()} if self.fixed else Q.loyo_pick({k: self.sc[f'{tag}_{k}'] for k in pts}, list(pts)))
        self.sel[f'{tag}_pick'] = {str(k): v for k, v in pk.items()}
        return self.add(tag, Q.loyo_series(pk, {k: self.fc[f'{tag}_{k}'] for k in pts}, self.year), True, f'LOYO over {tag} points')

    def best(self, key, names):
        if not self.fixed:
            self.sel[key] = min(names, key=lambda n: self.sc[n]['mean_QLIKE'])
        return self.sel[key]


def chain(a, inp, y, end, aux, ev):
    """inp: fin, hari (levels), bss (B** level), g1 (dict c-tag -> level), g2 (level). aux: z2 (n,2), stress (bool), z3 (n,3), rec."""
    rec = aux.get('rec')
    g1 = ev.choose('G1', inp['g1'])
    ev.add('G2', inp['g2'], True, 'B** with ln HAR_i, ln HAR-X+ as features')
    lns = np.log(np.column_stack([inp['fin'], inp['bss'], inp['hari'], g1]))
    ev.add('G3', Y3.a1_forecast(a, lns, y, end, H, g1, rec=rec), True, 'NNLS ridge alpha=1 over FIN_PIT, B**, HAR_i, G1')
    hard, calm, strs = g4_series(a, lns, y, end, aux['stress'], g1, rec)
    ev.add('G4_hard', hard, False)
    ev.add('G4_logit_calm', calm, False)
    ev.add('G4_logit_stress', strs, False)
    ev.add('G4_logit', Y3.a2_forecast(a, aux['z2'], y, end, H, calm, strs, rec=rec), False)
    pk = ({int(k): v for k, v in ev.sel['G4_pick'].items()} if ev.fixed else Q.loyo_pick({k: ev.sc[k] for k in ('G4_hard', 'G4_logit')}, ['G4_hard', 'G4_logit']))
    ev.sel['G4_pick'] = {str(k): v for k, v in pk.items()}
    ev.add('G4', Q.loyo_series(pk, {k: ev.fc[k] for k in ('G4_hard', 'G4_logit')}, a.year.values), True, 'LOYO hard / logistic')
    b5 = ev.best('base_G5', ['G1', 'G2', 'G3', 'G4'])
    ev.choose('G5', g5_points(a, ev.fc[b5], y))
    b6 = ev.best('base_G6', ['G1', 'G2', 'G3', 'G4', 'G5'])
    ev.add('G6', g6_series(a, ev.fc[b6], y, end, aux['z3'], rec), True, f'on {b6}')
    ev.best('G7_of', ['G1', 'G2', 'G3', 'G4', 'G5', 'G6'])
    return ev
