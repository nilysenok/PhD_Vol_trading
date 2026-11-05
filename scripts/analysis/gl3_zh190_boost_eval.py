# -*- coding: utf-8 -*-
"""Zh-190 stage 1: cells on disk, alignment checks, O3 metrics against the etalons (gl3_r6b_common)."""
import numpy as np
import pandas as pd
import gl3_r6b_common as C
import gl3_zh190_boost_core as K
from gl3_r56_diagnostics import read, REG


ETALONS = ('FIN_PIT', 'HARJb', 'HARi_LOYO')


def alignment_ok(b, h, feature_d):
    """Target of row d = mean of rv over d+1..d+h of the same ticker; the 'rv_d' feature of row d = rv of day d."""
    g = b.groupby('ticker').rv
    y = np.column_stack([g.shift(-j) for j in range(1, h + 1)]).mean(axis=1)
    t = b[f'_y{h}'].values
    ok = np.isfinite(y) & np.isfinite(t)
    f = np.asarray(feature_d, float)
    okf = np.isfinite(f)
    return bool(ok.sum() > 100 and np.allclose(y[ok], t[ok], rtol=1e-9, atol=1e-14) and np.allclose(f[okf], b.rv.values[okf], rtol=1e-9, atol=1e-14))


def cell_path(h, year, cid):
    return K.OUT / 'cells' / f'h{h}_{year}_{cid}.parquet'


def write_cell(h, year, cid, c, f):
    p = cell_path(h, year, cid)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(dict(date=c.b.date.values[f['pw']], ticker=c.tk[f['pw']], raw=f['raw'].astype('float32'),
                           s2=f['s2'].astype('float32'), corr=f['corr'].astype('float32'), pred=f['pred'].astype('float32')))
    df.to_parquet(p.with_suffix('.tmp'), index=False)
    p.with_suffix('.tmp').replace(p)


def read_cells(h, cid, years=range(2016, 2022)):
    return pd.concat([pd.read_parquet(cell_path(h, y, cid)) for y in years], ignore_index=True)


class Scorer:
    """Dev evaluation rows of one horizon (predictions_h{h}.parquet of round 3, 2017-2021) and the O3 metrics."""
    def __init__(self, data, h):
        self.h, self.data = h, data
        self.ref, self.idx = data.refs[h]
        r = self.ref
        self.y, self.year, self.tk = r.rv_actual.values, r.year.values, r.ticker.values
        self.m2020 = r.date.between('2020-03-01', '2020-06-30').values
        self.mrvi = data.frames['original']._stress.values[self.idx].astype(bool)
        self.key = pd.MultiIndex.from_frame(r[['date', 'ticker']])

    def align(self, df, col='pred'):
        ix = pd.MultiIndex.from_frame(df[['date', 'ticker']]).get_indexer(self.key)
        assert (ix >= 0).all(), 'forecast does not cover every dev row'
        return df[col].values[ix].astype('float64')

    def year_means(self, df, years):
        """Annual QLIKE means of the rows of `years` only (screening on 2019-2020 does not cover the other years)."""
        ix = pd.MultiIndex.from_frame(df[['date', 'ticker']]).get_indexer(self.key)
        out = {}
        for y in years:
            m = self.year == y
            assert (ix[m] >= 0).all(), f'cell does not cover {y}'
            out[y] = float(C.qlike_terms(self.y[m], df.pred.values[ix[m]].astype('float64')).mean())
        return out

    def score(self, p):
        return C.score(self.y, p, self.year, self.m2020, self.mrvi)

    def tickers(self, p):
        return C.ticker_means(self.y, p, self.year, self.tk)

    def etalons(self):
        e = {'HARJb': self.ref.har_corr.values.astype('float64')}
        if self.h == 22:
            e['FIN_PIT'] = e['HARJb']
        if self.h == 5:
            e['FIN_PIT'] = self.align(read(REG / 'gl3_r6b_h5/pit/experts/FIN_PIT.parquet', ['date', 'ticker', 'prediction']), 'prediction')
        d = K.OUT / 'etalony'
        for name in ETALONS:  # agent 2 (etalony/): FIN_PIT, HAR-J(b), HAR_i with LOYO lambda; they replace the built-in ones of the same name
            p = d / f'{name}_h{self.h}.parquet'
            if p.exists():
                e[name] = self.align(pd.read_parquet(p), 'prediction')
        return e

    def row(self, p, etal):
        s = self.score(p)
        out = {**{f'QLIKE_{y}': s[str(y)] for y in C.YEARS}, 'mean_QLIKE': s['mean_QLIKE'], 'mean_LOYO': s['mean_QLIKE'],
               'stress2020': s['stress2020'], 'stressRVI': s['stressRVI'],
               'ratio_stress2020': s['ratio_stress2020'], 'ratio_stressRVI': s['ratio_stressRVI']}
        mine = self.tickers(p)
        for name, q in etal.items():
            k, n = C.tickers_better(mine, self.tickers(q))
            out[f'tickers_better_vs_{name}'], out[f'binom_p_vs_{name}'] = k, C.binom_sf(k, n)
            out[f'delta_mean_vs_{name}'] = s['mean_QLIKE'] - self.score(q)['mean_QLIKE']
        return out
