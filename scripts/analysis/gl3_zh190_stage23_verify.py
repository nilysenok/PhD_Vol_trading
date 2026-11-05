# -*- coding: utf-8 -*-
"""Zh-190 stages 2-3 gates: G2 (ledger), G3 (independent recomputation of the metrics from the saved parquet, own numpy code),
G4 (rows, selection.json, attempt_counts.json). Usage: PYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_zh190_stage23_verify.py"""
import json
import numpy as np
import pandas as pd
import gl3_zh190_stage23_core as Q
from gl3_zh190_boost_checks import rvi_stress, REF, LAST
from gl3_zh190_boost_core import OUT as ST1_OUT

EXPECT = ['FIN_PIT', 'HARJb', 'HARi_LOYO', 'Bstar', *[f'{g}_{k}' for g in ('G1', 'G2') for k in ('c50', 'c75', 'c100', 'LOYO')], 'G3', 'Gstar', 'A1',
          'A2_HARi', 'A2_G1c100', 'A2', *[f'A3_hl{hl}_w{w}' for hl in (20, 60, 120) for w in (50, 75)], 'A3', 'Hstar']
ETAL = ('FIN_PIT', 'HARJb', 'HARi_LOYO')


def g2():
    ev = [json.loads(x) for x in Q.EVENTS.read_text().splitlines()]
    bad = [e for e in ev if e['h'] not in Q.HS or not (pd.Timestamp(e['max_target_end']) < pd.Timestamp(e['cutoff']) < pd.Timestamp('2022-01-01'))]
    return ev, dict(events=len(ev), bad=len(bad), horizons=sorted({e['h'] for e in ev}), ok=bool(ev and not bad and {e['h'] for e in ev} == set(Q.HS)))


def metrics(ref, p, stress):
    r = ref.rv_actual.values / p
    l = r - np.log(r) - 1
    ann = {y: l[ref.year.values == y].mean() for y in range(2017, 2022)}
    m20, hi = ref.date.between('2020-03-01', '2020-06-30').values, ref.date.map(stress).fillna(False).values.astype(bool)
    out = dict(mean_QLIKE=np.mean(list(ann.values())), stress2020=l[m20].mean(), stressRVI=l[hi].mean(), ratio_stress2020=r[m20].mean(), ratio_stressRVI=r[hi].mean(),
               **{f'QLIKE_{y}': v for y, v in ann.items()})
    return out, pd.Series(l).groupby([ref.ticker.values, ref.year.values]).mean().groupby(level=0).mean()


def g3():
    st, worst, count = rvi_stress(), 0., 0
    for h in Q.HS:
        ref = pd.read_parquet(str(REF).format(h=h), columns=['date', 'ticker', 'year', 'rv_actual'], filters=[('date', '<=', LAST)])
        ref = ref[ref.date >= '2017-01-01'].reset_index(drop=True)
        tab = pd.read_csv(Q.TAB / f'stage23_h{h}.csv').set_index('config')
        fc = pd.read_parquet(Q.OUT / f'forecasts_h{h}.parquet')
        assert fc.pred.dtype == np.float32 and fc.date.max() < pd.Timestamp('2022-01-01') and np.isfinite(fc.pred).all() and (fc.pred > 0).all()
        series = {n: g for n, g in fc.groupby('config')}

        def cover(df, col):
            m = ref[['date', 'ticker']].merge(df[['date', 'ticker', col]], on=['date', 'ticker'], how='left', validate='1:1')
            assert len(m) == len(ref) and m[col].notna().all(), 'coverage'
            return m[col].values.astype('float64')
        P = {n: cover(g, 'pred') for n, g in series.items()}
        for n in ETAL:
            P[n] = cover(pd.read_parquet(ST1_OUT / 'etalony' / f'{n}_h{h}.parquet'), 'prediction')
        M = {n: metrics(ref, p, st) for n, p in P.items()}
        for name, (mine, tm) in M.items():
            for key, v in mine.items():
                worst, count = max(worst, abs(float(tab.loc[name, key]) - v)), count + 1
            for e in ETAL:
                k = int((tm < M[e][1] - 1e-12).sum())
                assert k == tab.loc[name, f'tickers_better_vs_{e}'], (h, name, e)
    return dict(max_abs_diff=worst, values_compared=count, ok=bool(worst <= 1e-11 and count > 0))


def hari_consistent():
    """HAR_i level assembled from the lambda files of the LOYO picks equals the saved HARi_LOYO forecast (2017-2021)."""
    import gl3_zh190_stage23_fit as F
    worst = 0.
    for h in Q.HS:
        har, _, _ = F.load_hari(h)
        saved = pd.read_parquet(ST1_OUT / 'etalony' / f'HARi_LOYO_h{h}.parquet')
        m = saved.merge(har, on=['date', 'ticker'], how='left', validate='1:1')
        worst = max(worst, float(np.abs(m.level / m.prediction - 1).max()))
    return dict(max_rel_diff=worst, ok=bool(worst <= 1e-9))


def g4(ev):
    rows, sel = {}, json.loads((Q.TAB / 'selection.json').read_text())
    for h in Q.HS:
        t = pd.read_csv(Q.TAB / f'stage23_h{h}.csv')
        rows[str(h)] = len(t)
        assert set(t.config) == set(EXPECT) and len(t) == len(EXPECT), h
        assert {'Gstar', 'A2_pick', 'A3_pick', 'G1_pick', 'G2_pick', 'Hstar', 'lam_loyo'} <= set(sel[str(h)])
        hs = t[t.config == 'Hstar'].iloc[0]
        assert all(k in t.columns for k in ('itog_mean', 'itog_years', 'itog_stress', 'itog_tickers', 'itog_h')) and not pd.isna(hs.itog_h)
    f = pd.DataFrame(ev)
    fits = {str(h): {ph: int(((f.h == h) & (f.phase == ph)).sum()) for ph in sorted(f.phase.unique())} for h in Q.HS}
    cnt = dict(fits_by_h=fits, ledger_events=len(f), by_component=f.groupby('component').size().to_dict(), rows_by_h=rows,
               series_by_h={str(h): int(pd.read_parquet(Q.OUT / f'forecasts_h{h}.parquet').config.nunique()) for h in Q.HS},
               cells_fit_jobs={str(h): len(list((Q.OUT / 'cells').glob(f'h{h}_*_g?.parquet'))) for h in Q.HS})
    (Q.TAB / 'attempt_counts.json').write_text(json.dumps(cnt, indent=1, default=int))
    return dict(rows=rows, expected_rows_each=len(EXPECT), ok=all(v == len(EXPECT) for v in rows.values()), selection_keys=True)


def verify():
    ev, r2 = g2()
    out = dict(G2=r2, G3=g3(), hari=hari_consistent(), G4=g4(ev))
    (Q.TAB / 'verify.json').write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))
    assert all(v['ok'] for v in out.values())


if __name__ == '__main__':
    verify()
