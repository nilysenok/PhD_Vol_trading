# -*- coding: utf-8 -*-
"""Zh-190 stages 2-3, pure part (no data readers): ledger, G-1/G-2 levels, G-3 blend, LOYO, 'Itog h'.
Closed windows, EWMA and pooling are reused from round 6b (gl3_r6b_layers, with lag = h instead of 5)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import gl3_r6b_common as C
import gl3_r6b_layers as L

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / 'docs/tables/gl3_zh190/stage23'
OUT = ROOT / 'data/regen/gl3_zh190/stage23'
EVENTS = TAB / 'fit_events.jsonl'
HS = (1, 5, 22)
YEARS = C.YEARS
ALL_YEARS = (2016,) + YEARS            # 2016 = history for the layers, not scored (O1)
CS = (.5, .75, 1.)
SHRINK, MIN_OBS, WIN, EPS = .5, 30, 252, C.EPS


def ctag(c):
    return f'c{int(round(c * 100))}'


def f32(p):
    """Forecasts are stored and used as float32."""
    return np.asarray(p, float).astype('float32').astype('float64')


def level(z):
    return np.exp(np.clip(z, -60, 60))


def record(h, kind, cut, n, maxend, phase='fit', detail=None):
    """O10 ledger: max_target_end < cutoff < 2022-01-01, otherwise the fit is refused."""
    assert maxend is not None and pd.Timestamp(maxend) < pd.Timestamp(cut) < pd.Timestamp('2022-01-01'), ('target end >= cut', kind, cut, maxend)
    r = dict(component=kind, h=h, cutoff=str(cut), n=int(n), max_target_end=str(maxend), phase=phase, detail=detail or {}, time_utc=pd.Timestamp.now(tz='UTC').isoformat())
    TAB.mkdir(parents=True, exist_ok=True)
    with EVENTS.open('a') as f:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')


def past_mask(end, cut):
    """Rows whose target is fully known before the cut (end of target < cut)."""
    return np.asarray(end, 'datetime64[ns]') < np.datetime64(pd.Timestamp(cut))


def parse_config(cfg):
    p = cfg.split('-')
    return p[0], p[1], 'bag10' in p, 'opt' in p


def shrink_s2(s2_t, s2_pool):
    """Ticker variance shrunk by 0.5 to the common one."""
    return SHRINK * s2_t + (1 - SHRINK) * s2_pool


def val_s2(res, gv, tk, c):
    """G-1 correction input: variance (ddof 0) of the final log error ln RV - (ln HAR_i + c f) on the inner validation (O2), by ticker and pooled."""
    err = np.asarray(res, float) - c * np.asarray(gv, float)
    return pd.Series(err).groupby(np.asarray(tk)).var(ddof=0), float(np.var(err))


def g1_level(ln_har, f, c, s2):
    return level(np.asarray(ln_har) + c * np.asarray(f) + np.asarray(s2) / 2)


def g2_level(ln_har, f, c):
    return level(np.asarray(ln_har) + c * np.asarray(f))


def blend_ln(lvl_b, lvl_h, w):
    """G-3: ln sigma^2 = w ln B + (1-w) ln HAR_i (both already with their corrections)."""
    return np.exp(w * np.log(lvl_b) + (1 - w) * np.log(lvl_h))


def g3_weight(a, lvl_b, lvl_h, y, h):
    """w_i = Q^HAR_i / (Q^HAR_i + Q^B_i) over the last 252 closed windows of the ticker (target end <= d: lag = h), w_t the same for the pool,
    w = 0.5 w_i + 0.5 w_t clipped to [0; 1]; where nothing is closed yet w = 0.5."""
    ok = np.isfinite(lvl_b) & np.isfinite(lvl_h)
    qb = L.roll_mean(a, L.closed(a, L.qlike(L.ratio(y, np.where(ok, lvl_b, np.nan))), h), WIN)
    qh = L.roll_mean(a, L.closed(a, L.qlike(L.ratio(y, np.where(ok, lvl_h, np.nan))), h), WIN)
    wi, wt = qh / (qh + qb), L.date_pool(a, qh) / (L.date_pool(a, qh) + L.date_pool(a, qb))
    w = np.clip(.5 * np.where(np.isnan(wi), wt, wi) + .5 * wt, 0, 1)
    return np.where(np.isfinite(w), w, .5)


def loyo_pick(scores, names):
    """O5: the point of year Y is the one with the smallest mean QLIKE over the other four dev years; 2016 (history) takes the pick of 2017."""
    pick = {Y: min(names, key=lambda k: np.mean([scores[k][str(y)] for y in YEARS if y != Y])) for Y in YEARS}
    pick[2016] = pick[2017]
    return pick


def loyo_series(pick, preds, year):
    out = np.full(len(year), np.nan)
    for Y, k in pick.items():
        m = np.asarray(year) == Y
        out[m] = preds[k][m]
    return out


def closed_ok(date, end, ticker, h):
    """The row-shift 'closed window' (value of row j known at row j+h of the ticker) never uses a target that ends after the date of the row."""
    d, e, t = np.asarray(date, 'datetime64[ns]'), np.asarray(end, 'datetime64[ns]'), np.asarray(ticker)
    ok = True
    for tk in np.unique(t):
        m = t == tk
        dd, ee = d[m], e[m]
        ok &= bool((ee[:-h] <= dd[h:]).all()) if m.sum() > h else True
    return ok


def itog_h(hrow, base):
    """Itog h (Zh-190 p. 4): H* vs FIN_PIT and HAR_i LOYO: mean better than both, >= 3 of 5 years better than each, both stress rows not worse
    than each, >= 10 of 17 tickers better than each. One column per condition."""
    names = ('FIN_PIT', 'HARi_LOYO')
    yrs = {n: int(sum(hrow[f'QLIKE_{y}'] < base[n][f'QLIKE_{y}'] - EPS for y in YEARS)) for n in names}
    out = dict(itog_mean=all(hrow['mean_QLIKE'] < base[n]['mean_QLIKE'] - EPS for n in names),
               itog_years=all(yrs[n] >= 3 for n in names),
               itog_stress=all(hrow[k] <= base[n][k] + EPS for n in names for k in ('stress2020', 'stressRVI')),
               itog_tickers=all(hrow[f'tickers_better_vs_{n}'] >= 10 for n in names))
    out = {k: bool(v) for k, v in out.items()}
    return {**out, **{f'years_better_vs_{n}': yrs[n] for n in names}, 'itog_h': all(out.values())}
