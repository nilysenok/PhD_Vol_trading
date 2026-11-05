# -*- coding: utf-8 -*-
"""Zh-193 item B1: day-d features from 1-minute bars (2014-2021 only). Main session slice and calendar as K13 (gl3_r56_measure.one_day):
price available at begin+1 minute, session 09:50-18:50 (available minutes 591..1130); 5-minute grids shifted by 0..9 minutes, mean over the 10 grids."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BARS = ROOT / 'data/regen/D25/bars1m'
OUT = ROOT / 'data/regen/gl3_zh193_h1'
LAST_YEAR = 2021
S0, S1, STEP, HOUR = 590, 1130, 5, 60
EVE0, EVE1 = 1145, 1430
NEW = ['ln_rv5s', 'ln_bv5s', 'jump5s', 'ln_rv_h1st', 'ln_rv_hlast', 'ln_rv_eve_prev', 'gap_sig', 'ln_rs_pos', 'ln_rs_neg', 'ln_rq5s']


def _lg(v):
    return np.log(np.clip(v, 1e-12, None))


def grid_stats(t, p, offset, lo=S0, hi=S1):
    """Realized stats of one clock grid; t = available minutes (sorted), p = prices. None if fewer than 3 grid prices."""
    grid = np.arange(lo + offset, hi + 1, STEP)
    ix = np.searchsorted(t, grid, side='right') - 1
    ok = ix >= 0
    ix, g = ix[ok], grid[ok]
    if len(ix) < 3:
        return None
    r = np.diff(np.log(p[ix]))
    e = g[1:]
    n = len(r)
    return dict(rv=np.sum(r * r), bv=np.pi / 2 * np.sum(np.abs(r[1:] * r[:-1])), rsp=np.sum(r[r > 0] ** 2), rsn=np.sum(r[r < 0] ** 2),
                rq=n / 3 * np.sum(r ** 4), h1=np.sum(r[e <= lo + HOUR] ** 2), hl=np.sum(r[e > hi - HOUR] ** 2))


def day_features(t, p):
    """Main-session features of one day from available minutes t and prices p (price of the last minute-bar is p[-1]); None if too few points."""
    st = [grid_stats(t, p, o) for o in range(10)]
    if any(s is None for s in st):
        return None
    m = {k: float(np.mean([s[k] for s in st])) for k in st[0]}
    return dict(rv5s=m['rv'], bv5s=m['bv'], jump5s=max(m['rv'] - m['bv'], 0.), rv_h1st=m['h1'], rv_hlast=m['hl'], rs_pos=m['rsp'], rs_neg=m['rsn'], rq5s=m['rq'])


def evening_rv(t, p):
    s = grid_stats(t, p, 0, EVE0, EVE1)
    return np.nan if s is None else float(s['rv'])


def read_bars(path):
    """Guard: only years <= 2021 are opened; the file name is checked before reading, timestamps after."""
    path = Path(path)
    year = int(path.stem.rsplit('_', 1)[1])
    if year > LAST_YEAR:
        raise ValueError(f'bars of {year} are closed (data >= 2022 are not read)')
    df = pd.read_csv(path, usecols=['begin', 'open', 'close'])
    df = df.drop_duplicates('begin', keep='last').sort_values('begin')
    ts = pd.to_datetime(df.begin)
    assert ts.dt.year.eq(year).all() and ts.max() < pd.Timestamp('2022-01-01')
    df['date'] = ts.dt.normalize()
    df['avail'] = (ts.dt.hour * 60 + ts.dt.minute + 1).values
    return df[df.close > 0]


def ticker_days(ticker):
    rows = []
    for y in range(2014, LAST_YEAR + 1):
        for date, g in read_bars(BARS / f'{ticker}_{y}.csv').groupby('date', sort=True):
            main = g[(g.avail >= S0 + 1) & (g.avail <= S1)]
            if len(main) < 3:
                continue
            f = day_features(main.avail.values, main.close.values)
            if f is None:
                continue
            ev = g[(g.avail >= EVE0) & (g.avail <= EVE1)]
            eve = evening_rv(ev.avail.values, ev.close.values) if len(ev) >= 3 else np.nan
            rows.append(dict(date=date, ticker=ticker, open_d=float(main.open.values[0]), close_d=float(main.close.values[-1]), rv_eve=eve, **f))
    return pd.DataFrame(rows)


def build(tickers):
    """Frame (date, ticker, raw day features); previous-day columns are built in `assemble`."""
    return pd.concat([ticker_days(t) for t in tickers], ignore_index=True)


def assemble(raw, b):
    """New columns on the rows of the frame b (sorted by ticker, date): all known at the close of the main session of d.
    gap = ln(open_d / close_{d-1}) / sqrt(RV_w) with RV_w from b; evening RV of the previous trading day of the ticker."""
    m = b[['date', 'ticker', 'rv_w']].merge(raw, on=['date', 'ticker'], how='left', validate='1:1')
    g = m.groupby('ticker')
    gap = np.log(m.open_d / g.close_d.shift(1))
    out = pd.DataFrame({'ln_rv5s': _lg(m.rv5s), 'ln_bv5s': _lg(m.bv5s), 'jump5s': m.jump5s, 'ln_rv_h1st': _lg(m.rv_h1st), 'ln_rv_hlast': _lg(m.rv_hlast),
                        'ln_rv_eve_prev': _lg(g.rv_eve.shift(1)),
                        'gap_sig': gap / np.sqrt(m.rv_w.clip(lower=1e-12)), 'ln_rs_pos': _lg(m.rs_pos), 'ln_rs_neg': _lg(m.rs_neg), 'ln_rq5s': _lg(m.rq5s)})
    out = out[NEW]
    out[~m.rv5s.notna().values] = np.nan
    return out.astype('float32')
