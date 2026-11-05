# -*- coding: utf-8 -*-
"""Zh-194 (h = 5, 22), pure part: ledger, G6 (variance-regression multiplier), EWMA points, pick objects (best / LOYO, or forced picks of the seed-42 run).
Closed window: a forecast made not later than d-h (lag = h); an annual cutoff uses rows whose target ends before the cut (PIT)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import gl3_r6b_layers as L
import gl3_zh190_stage23_core as Q

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / 'docs/tables/gl3_zh194'
OUT = ROOT / 'data/regen/gl3_zh194'
EVENTS = TAB / 'fit_events.jsonl'
HS = (5, 22)
SEEDS = (42, 43, 44, 45, 46)
HALF_LIVES, WS = (10, 20, 60), (.5, .75)
G6_FEATS = ('close_RVI', 'rq', 'rv_w')   # RVI_d, RQ_d, RV_w of the row's own day d


def record(h, kind, cut, n, maxend, phase='fit', detail=None):
    """O10 ledger of the layers: h in {5, 22}, max_target_end < cutoff < 2022-01-01, otherwise the fit is refused."""
    assert h in HS and maxend is not None and pd.Timestamp(maxend) < pd.Timestamp(cut) < pd.Timestamp('2022-01-01'), ('target end >= cut', kind, cut, maxend)
    r = dict(component=kind, h=h, cutoff=str(cut), n=int(n), max_target_end=str(maxend), phase=phase, detail=detail or {}, time_utc=pd.Timestamp.now(tz='UTC').isoformat())
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS.open('a') as f:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')


def g6_features(b):
    """ln RVI_d, ln RQ_d, ln RV_w of day d (known at the close of d), NaN-safe."""
    return np.column_stack([np.log(np.clip(b[c].values.astype(float), 1e-12, None)) for c in G6_FEATS])


def g6_forecast(a, base, y, end, h, feats, rec=None):
    """G6: base * exp(v/2), v = max(0, [1, feats] @ beta); beta = OLS of the squared log error of the base on [1, feats] over rows with target end < cut
    (all past years from 2016, closed windows), annual cutoff. 2016 has no history: base."""
    base = np.asarray(base, float)
    out, ly = base.copy(), np.log(np.clip(y, 1e-12, None))
    e2 = (ly - np.log(base)) ** 2
    X = np.column_stack([np.ones(len(a)), feats])
    for Y in Q.YEARS:
        cut = pd.Timestamp(f'{Y}-01-01')
        tr = Q.past_mask(end, cut) & np.isfinite(e2) & (y > 1e-12) & np.isfinite(X).all(axis=1)
        if tr.sum() < Q.MIN_OBS:
            continue
        beta = np.linalg.lstsq(X[tr], e2[tr], rcond=None)[0]
        va = (a.year.values == Y) & np.isfinite(X).all(axis=1)
        with np.errstate(all='ignore'):   # spurious BLAS (Accelerate) matmul warnings on finite data, as in 6b
            out[va] = base[va] * np.exp(np.maximum(X[va] @ beta, 0.) / 2)
        if rec:
            rec(h, 'G6_ols', cut, int(tr.sum()), np.asarray(end)[tr].max(), detail=dict(beta=beta.tolist()))
    return out


def ewma_points(a, base, y, h):
    """(v) k_i,t = EWMA(RV/sigma^2) over closed windows (lag h), half-life {10, 20, 60}, w in {0.5, 0.75}, k in [0.7; 1.5]."""
    return {f'hl{hl}_w{int(w * 100)}': L.layer14(a, base, y, hl, w, lag=h) for hl in HALF_LIVES for w in WS}


class Pick:
    """Choice by the mean QLIKE (best) or the LOYO point (O5). `forced` = the choices recorded by the seed-42 run: other seeds repeat the structure, not the choice."""
    def __init__(self, score=None, forced=None):
        assert (score is None) != (forced is None)
        self.score, self.forced, self.log = score, forced, {}

    def best(self, tag, series):
        name = self.forced[tag] if self.forced else min(series, key=lambda k: self.score(series[k])['mean_QLIKE'])
        self.log[tag] = name
        return name

    def loyo(self, tag, preds, year):
        if self.forced:
            pk = {int(k): v for k, v in self.forced[tag].items()}
        else:
            sc = {k: self.score(v) for k, v in preds.items()}
            pk = Q.loyo_pick(sc, list(preds))
        self.log[tag] = {str(k): v for k, v in pk.items()}
        return Q.loyo_series(pk, preds, year)
