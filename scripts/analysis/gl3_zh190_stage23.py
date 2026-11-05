# -*- coding: utf-8 -*-
"""Zh-190 stages 2-3 (h = 1, 5, 22 in one run): G-1, G-2, G-3, G*, A-1, A-2, A-3, H*, 'Itog h'.
Usage: PYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_zh190_stage23.py <fit|layers|verify|all>"""
# xtask: файл длиннее 200 строк — драйвер одного прогона (сборка рядов по h, таблицы, выбор)
import json
import sys
import time
import numpy as np
import pandas as pd
import gl3_zh190_boost_core as K
import gl3_zh190_boost_eval as E
import gl3_zh190_stage23_core as Q
import gl3_zh190_stage23_fit as F
import gl3_zh190_stage23_layers as Y3


def log(msg):
    with (Q.TAB / 'run.log').open('a') as f:
        f.write(f'{pd.Timestamp.now(tz="UTC").isoformat()} {msg}\n')


def on_a(key, df, col):
    """Values of df[col] on the rows of the frame `a` (index `key`); every row must be covered, finite."""
    out = pd.Series(df[col].values, index=pd.MultiIndex.from_frame(df[['date', 'ticker']])).reindex(key).values.astype(float)
    assert np.isfinite(out).all(), ('forecast does not cover every row', col)
    return out


def read_cells(h, kind, a_key):
    """G-1 / G-2 cells of all cutoffs 2016-2021: (date, ticker, ln_har, f) and the validation variances. QLIKE B*: G-2 = the shared fit."""
    k = 'g1' if kind == 'g2' and Q.parse_config(F.G['bstar'][h])[1] == 'qlike' else kind
    cells = pd.concat([pd.read_parquet(F.cell_path(h, Y, k)) for Y in Q.ALL_YEARS], ignore_index=True)
    val = pd.concat([pd.read_parquet(F.cell_path(h, Y, k, '_val')).assign(year=Y) for Y in Q.ALL_YEARS], ignore_index=True)
    return cells, val


def g1_points(cells, val, key):
    cells = cells.assign(year=cells.date.dt.year)
    out = {}
    for c in Q.CS:
        v = val[np.isclose(val.c, c)]
        m = cells.merge(v[['year', 'ticker', 's2_t']], on=['year', 'ticker'], how='left', validate='m:1')
        pool = m.year.map(v.groupby('year').s2_pool.first())
        s2 = Q.shrink_s2(m.s2_t.fillna(pool).values, pool.values)
        out[Q.ctag(c)] = on_a(key, m.assign(p=Q.g1_level(m.ln_har.values, m.f.values, c, s2)), 'p')
    return out


def g2_points(cells, key):
    return {Q.ctag(c): on_a(key, cells.assign(p=Q.g2_level(cells.ln_har.values, cells.f.values, c)), 'p') for c in Q.CS}


def best(scores, names):
    return min(names, key=lambda k: scores[k]['mean_QLIKE'])


def build_h(d, h):
    b = d.frames['original']
    m16 = (b.date >= '2016-01-01').values
    a = b[m16].reset_index(drop=True)
    y, end, year = a[f'_y{h}'].values.astype(float), a[f'_end{h}'].values, a.year.values
    key = pd.MultiIndex.from_frame(a[['date', 'ticker']])
    sc = E.Scorer(d, h)
    ix = key.get_indexer(sc.key)
    assert (ix >= 0).all()
    etal = sc.etalons()
    rows, fc, scores, sel = [], {}, {}, dict(h=h, closed_window_ok=Q.closed_ok(a.date, end, a.ticker, h))

    def add(name, p, stage, eligible, note=''):
        p = Q.f32(p)
        assert np.isfinite(p).all() and (p > 0).all(), name
        fc[name] = p
        r = sc.row(p[ix], etal)
        if not eligible:
            r['mean_LOYO'] = np.nan
        rows.append(dict(h=h, stage=stage, config=name, eligible=eligible, note=note, **r))
        scores[name] = sc.score(p[ix])
        return scores[name]

    har, pick, _ = F.G['hari'][h]
    har_lvl = Q.f32(on_a(key, har, 'level'))
    bst_cfg = F.G['bstar'][h]
    bst = Q.f32(on_a(key, pd.read_parquet(Q.ROOT / f'data/regen/gl3_zh190/bstar_h{h}.parquet'), 'pred'))
    for n, p in etal.items():
        rows.append(dict(h=h, stage='etalon', config=n, eligible=True, note='', **sc.row(p, etal)))
    base_rows = {r['config']: r for r in rows}
    add('Bstar', bst, 'stage1', True, bst_cfg)
    sel['lam_loyo'] = {str(k): v for k, v in pick.items()}
    # stage 2
    cells1, val1 = read_cells(h, 'g1', key)
    cells2, _ = read_cells(h, 'g2', key)
    for tag, pts in (('G1', g1_points(cells1, val1, key)), ('G2', g2_points(cells2, key))):
        pts = {k: Q.f32(v) for k, v in pts.items()}
        for k, v in pts.items():
            add(f'{tag}_{k}', v, 'stage2', False)
        pk = Q.loyo_pick({k: scores[f'{tag}_{k}'] for k in pts}, list(pts))
        add(f'{tag}_LOYO', Q.loyo_series(pk, pts, year), 'stage2', True, 'LOYO over c; 2016 takes the pick of 2017')
        sel[f'{tag}_pick'] = {str(k): v for k, v in pk.items()}
    w = Q.g3_weight(a, bst, har_lvl, y, h)
    add('G3', Q.blend_ln(bst, har_lvl, w), 'stage2', True, 'daily weights, 252 closed windows')
    gname = best(scores, ('G1_LOYO', 'G2_LOYO', 'G3'))
    add('Gstar', fc[gname], 'stage2', True, f'= {gname}')
    sel['Gstar'] = gname
    # stage 3
    s1 = pd.read_csv(K.TAB / 'stage1_summary.csv')
    s1 = s1[(s1.h == h) & (s1.stage != 'etalon')]
    allfc = pd.read_parquet(K.OUT / f'forecasts_h{h}.parquet')
    legs = {}
    for leg, schemes in (('panel', ('panel', 'cat')), ('ticker', ('ticker', 'cont'))):
        cand = s1[s1.scheme.isin(schemes)].sort_values('mean_QLIKE', kind='stable')
        if cand.empty:  # h = 22: no ticker / cont row in stage 1 (only panel and cat passed the screen) - the leg is absent, recorded as a deviation
            sel[f'a1_{leg}'] = 'absent in stage 1'
            continue
        r = cand.iloc[0]
        legs[leg] = Q.f32(on_a(key, allfc[(allfc.stage == r.stage) & (allfc.config == r.config)], 'pred'))
        sel[f'a1_{leg}'] = f'{r.stage}/{r.config}'
    gs = fc['Gstar']
    lns = np.log(np.column_stack([har_lvl, *legs.values(), gs]))
    a1 = Y3.a1_forecast(a, lns, y, end, h, gs, rec=Q.record)
    add('A1', a1, 'stage3', True, 'ridge alpha=1, non-negative, shrink 0.5 to pool; 2016 = G*')
    z = Y3.z_features(b)[m16]
    st = {'A2_HARi': har_lvl, 'A2_G1c100': fc['G1_c100']}
    for n, s in st.items():
        add(n, Y3.a2_forecast(a, z, y, end, h, gs, s, rec=Q.record), 'stage3', False, 'stress expert ' + n[3:])
    pk = Q.loyo_pick({n: scores[n] for n in st}, list(st))
    add('A2', Q.loyo_series(pk, {n: fc[n] for n in st}, year), 'stage3', True, 'LOYO between the two stress experts')
    sel['A2_pick'] = {str(k): v for k, v in pk.items()}
    base_n = best(scores, ('Gstar', 'A1', 'A2'))
    pts = {k: Q.f32(v) for k, v in Y3.a3_points(a, fc[base_n], y, h).items()}
    for k, v in pts.items():
        add(f'A3_{k}', v, 'stage3', False, f'on {base_n}')
    pk = Q.loyo_pick({k: scores[f'A3_{k}'] for k in pts}, list(pts))
    add('A3', Q.loyo_series(pk, pts, year), 'stage3', True, f'LOYO over 6 points on {base_n}')
    sel.update(A3_base=base_n, A3_pick={str(k): v for k, v in pk.items()})
    hn = best(scores, ('Gstar', 'A1', 'A2', 'A3'))
    add('Hstar', fc[hn], 'stage3', True, f'= {hn}' + (f' = {gname}' if hn == 'Gstar' else ''))
    sel['Hstar'] = hn
    rows[-1].update(Q.itog_h(rows[-1], base_rows))
    return pd.DataFrame(rows), fc, a, sel


def write_h(h, tab, fc, a, sel):
    Q.TAB.mkdir(parents=True, exist_ok=True)
    tab.to_csv(Q.TAB / f'stage23_h{h}.csv', index=False)
    f = pd.concat([pd.DataFrame(dict(config=n, date=a.date.values, ticker=a.ticker.values, pred=p.astype('float32'))) for n, p in fc.items()], ignore_index=True)
    f.to_parquet(Q.OUT / f'forecasts_h{h}.parquet', index=False)
    return sel


def summary(tabs):
    out = []
    for t in tabs:
        g, hs = t[t.config == 'Gstar'].iloc[0], t[t.config == 'Hstar'].iloc[0]
        out.append(dict(h=g.h, gstar=g.note, gstar_mean=g.mean_QLIKE, hstar=hs.note, hstar_mean=hs.mean_QLIKE, bstar_mean=t[t.config == 'Bstar'].mean_QLIKE.iloc[0],
                        **{k: hs[k] for k in hs.index if k.startswith(('QLIKE_', 'stress', 'ratio_', 'tickers_better', 'binom', 'itog', 'years_better'))}))
    return pd.DataFrame(out)


def layers():
    d = F.setup()
    tabs, sels = [], {}
    for h in Q.HS:
        t0 = time.time()
        tab, fc, a, sel = build_h(d, h)
        sels[str(h)] = write_h(h, tab, fc, a, sel)
        tabs.append(tab)
        log(f'layers h{h}: {len(tab)} rows, {len(fc)} series, {time.time() - t0:.0f}s')
    summary(tabs).to_csv(Q.TAB / 'stage23_summary.csv', index=False)
    (Q.TAB / 'selection.json').write_text(json.dumps(sels, indent=1, default=str))


def fit():
    d = F.setup()
    jobs = F.jobs_for()
    log(f'fit {len(jobs)} jobs, {F.run_pool(jobs, Q.TAB / "run.log"):.0f}s')


if __name__ == '__main__':
    stage = sys.argv[1]
    if stage in ('fit', 'all'):
        fit()
    if stage in ('layers', 'all'):
        layers()
    if stage in ('verify', 'all'):
        import gl3_zh190_stage23_verify as V
        V.verify()
