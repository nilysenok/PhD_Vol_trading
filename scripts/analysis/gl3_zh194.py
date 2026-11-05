# -*- coding: utf-8 -*-
"""Zh-194 (h = 5 and h = 22 in one run): stress expert and layers over H*; tables rows_h5.csv, rows_h22.csv, summary.csv.
Usage: PYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_zh194.py <fit|tables|verify|all>"""
import json
import sys
import time
import numpy as np
import pandas as pd
import gl3_r6b_common as C
import gl3_zh190_boost_eval as E
import gl3_zh190_stage23 as S23
import gl3_zh190_stage23_core as Q
import gl3_zh190_stage23_layers as Y3
import gl3_zh194_chain as CH
import gl3_zh194_core as Z
import gl3_zh194_fit as Fit

ETN = ('FIN_PIT', 'HARJb', 'HARi_LOYO')


def log(msg):
    with (Z.TAB / 'run.log').open('a') as f:
        f.write(f'{pd.Timestamp.now(tz="UTC").isoformat()} {msg}\n')


def harj_on(key, h):
    f = pd.read_parquet(Z.ROOT / f'data/regen/gl3_zh190/etalony/HARJb_h{h}.parquet')
    return pd.Series(f.prediction.values.astype(float), index=pd.MultiIndex.from_frame(f[['date', 'ticker']])).reindex(key).values


def build(d, h):
    b = d.frames['original']
    m16 = (b.date >= '2016-01-01').values
    a = b[m16].reset_index(drop=True)
    y, end = a[f'_y{h}'].values.astype(float), a[f'_end{h}'].values
    key = pd.MultiIndex.from_frame(a[['date', 'ticker']])
    sc = E.Scorer(d, h)
    ix = key.get_indexer(sc.key)
    assert (ix >= 0).all()
    return b, a, y, end, key, sc, ix, sc.etalons()


def table_rows(h, bk, fn, extra, sc, ix, etal):
    """O3 rows (as stage 2-3) + years better vs each etalon + the `cel` column of the Zh-194 section."""
    rows = [dict(h=h, stage='etalon', config=n, eligible=True, note='', **sc.row(p, etal)) for n, p in etal.items()]
    ref = {r['config']: r for r in rows}
    series = [(n, p, st, el, nt) for n, (p, st, el, nt) in bk.S.items()] + extra
    for n, p, st, el, nt in series:
        r = sc.row(p[ix], etal)
        if not el:
            r['mean_LOYO'] = np.nan
        rows.append(dict(h=h, stage=st, config=n, eligible=el, note=nt, **r))
    ref_name = 'Hstar' if h == 5 else 'G1_LOYO'
    ref_mean, harj_s = next(r for r in rows if r['config'] == ref_name)['mean_QLIKE'], ref['HARJb']['stress2020']
    for r in rows:
        for n in ETN:
            r[f'years_better_vs_{n}'] = int(sum(r[f'QLIKE_{y}'] < ref[n][f'QLIKE_{y}'] - C.EPS for y in C.YEARS))
        r['cel'] = bool(r['stress2020'] <= harj_s + C.EPS and r['mean_QLIKE'] <= ref_mean + C.EPS)
    return rows


def run_h(d, h):
    b, a, y, end, key, sc, ix, etal = build(d, h)
    z, feats = Y3.z_features(b)[(b.date >= '2016-01-01').values], Z.g6_features(a)
    har = Q.f32(S23.on_a(key, Fit.F.G['hari'][h][0], 'level'))
    comps = {s: {**Fit.read_comp(h, s, key, S23), 'har': har, 'harj': harj_on(key, h)} for s in Z.SEEDS}
    pk = Z.Pick(score=lambda p: sc.score(p[ix]))
    bk, fn = CH.chain(h, a, z, feats, y, end, comps[42], pk)
    forced, fin, extra = dict(pk.log), {42: bk[fn]}, []
    for s in Z.SEEDS[1:]:
        bs, fs = CH.chain(h, a, z, feats, y, end, comps[s], Z.Pick(forced=forced))
        assert fs == fn
        fin[s] = bs[fn]
        extra.append((f'seed{s}', fin[s], 'seed', False, f'chain of seed {s} on the structure of seed 42, final = {fn}'))
    dd = Q.f32(np.mean([fin[s] for s in Z.SEEDS], axis=0))
    extra.append(('D', dd, 'layer_d', True, f'mean in levels over seeds 42-46 of {fn}'))
    rows = pd.DataFrame(table_rows(h, bk, fn, extra, sc, ix, etal))
    f = pd.concat([pd.DataFrame(dict(config=n, date=a.date.values, ticker=a.ticker.values, pred=p.astype('float32')))
                   for n, p in [(n, v[0]) for n, v in bk.S.items()] + [(e[0], e[1]) for e in extra]], ignore_index=True)
    return rows, f, dict(h=h, final_before_seeds=fn, picks=forced, closed_window_ok=Q.closed_ok(a.date, end, a.ticker, h), seeds=list(Z.SEEDS))


def summary(tabs):
    keep = []
    for t in tabs:
        names = ['Hstar', 'a', 'b', 'V', 'G6', 'D'] if t.h.iloc[0] == 5 else ['G1_LOYO', 'A2_22', 'V', 'G6', 'D']
        keep.append(t[t.stage.eq('etalon') | t.config.isin(names)])
    return pd.concat(keep, ignore_index=True)


def tables():
    d = Fit.setup()
    tabs, sels = [], {}
    for h in Z.HS:
        t0 = time.time()
        rows, f, sel = run_h(d, h)
        rows.to_csv(Z.TAB / f'rows_h{h}.csv', index=False)
        f.to_parquet(Z.OUT / f'forecasts_h{h}.parquet', index=False)
        tabs.append(rows)
        sels[str(h)] = sel
        log(f'tables h{h}: {len(rows)} rows, {f.config.nunique()} series, {time.time() - t0:.0f}s')
    summary(tabs).to_csv(Z.TAB / 'summary.csv', index=False)
    (Z.TAB / 'selection.json').write_text(json.dumps(sels, indent=1, default=str))


if __name__ == '__main__':
    stage = sys.argv[1]
    if stage in ('fit', 'all'):
        Fit.setup()
        log(f'fit {len(Fit.jobs())} jobs, {Fit.run_pool():.0f}s')
    if stage in ('tables', 'all'):
        tables()
    if stage in ('verify', 'all'):
        import gl3_zh194_verify as V
        V.verify()
