# -*- coding: utf-8 -*-
"""Zh-194 gates: G2 (control: the rebuilt chain of seed 42 = stage-3 series, <= 1e-9 rel), G3 (ledger), G4 (independent recomputation of the metrics
from the saved parquet with own numpy code, <= 1e-11; forecasts finite, > 0, date < 2022, cover every dev row of the etalon), G5 (rows, selection.json, attempt_counts.json).
Usage: PYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_zh194_verify.py"""
import json
import numpy as np
import pandas as pd
import gl3_zh190_stage23_verify as V23
import gl3_zh194_core as Z
from gl3_zh190_boost_checks import rvi_stress, REF, LAST

ETAL = V23.ETAL
LAYERS = [f'V_hl{hl}_w{w}' for hl in Z.HALF_LIVES for w in (50, 75)] + ['V', 'G6', 'D'] + [f'seed{s}' for s in Z.SEEDS[1:]]
NEED = {5: ['Bstar', 'G3', 'Gstar', 'Hstar', 'a', 'b_stress', 'b', 'G1_c100'] + LAYERS, 22: ['G1_c100', 'G1_LOYO', 'A2_22'] + LAYERS}
S23_SAME = {5: ['Bstar', 'G1_c50', 'G1_c75', 'G1_c100', 'G1_LOYO', 'G2_LOYO', 'G3', 'Gstar', 'A2_HARi', 'A2_G1c100', 'Hstar'],
            22: ['G1_c50', 'G1_c75', 'G1_c100', 'G1_LOYO']}


def g2():
    """Control: series rebuilt by this code (retrained seed 42) against the stored stage-3 series (H*_h5 = A-2 with G-1 c=1 among them)."""
    out, worst = {}, 0.
    for h in Z.HS:
        mine = pd.read_parquet(Z.OUT / f'forecasts_h{h}.parquet')
        old = pd.read_parquet(Z.ROOT / f'data/regen/gl3_zh190/stage23/forecasts_h{h}.parquet')
        for n in S23_SAME[h]:
            m = mine[mine.config == n].merge(old[old.config == n], on=['date', 'ticker'], validate='1:1', suffixes=('', '_old'))
            assert len(m) == (mine.config == n).sum() == (old.config == n).sum(), (h, n)
            r = float(np.abs(m.pred.astype('float64') / m.pred_old.astype('float64') - 1).max())
            out[f'h{h}_{n}'] = r
            worst = max(worst, r)
    return dict(max_rel_diff=worst, series=out, ok=bool(worst <= 1e-9))


def g3():
    ev = [json.loads(x) for x in Z.EVENTS.read_text().splitlines()]
    bad = [e for e in ev if e['h'] not in Z.HS or not (pd.Timestamp(e['max_target_end']) < pd.Timestamp(e['cutoff']) < pd.Timestamp('2022-01-01'))]
    return ev, dict(events=len(ev), bad=len(bad), horizons=sorted({e['h'] for e in ev}), ok=bool(ev and not bad and {e['h'] for e in ev} == set(Z.HS)))


def g4():
    st, worst, count = rvi_stress(), 0., 0
    for h in Z.HS:
        ref = pd.read_parquet(str(REF).format(h=h), columns=['date', 'ticker', 'year', 'rv_actual'], filters=[('date', '<=', LAST)])
        ref = ref[ref.date >= '2017-01-01'].reset_index(drop=True)
        tab = pd.read_csv(Z.TAB / f'rows_h{h}.csv').set_index('config')
        fc = pd.read_parquet(Z.OUT / f'forecasts_h{h}.parquet')
        assert fc.pred.dtype == np.float32 and fc.date.max() < pd.Timestamp('2022-01-01') and np.isfinite(fc.pred).all() and (fc.pred > 0).all()

        def cover(df, col):
            m = ref[['date', 'ticker']].merge(df[['date', 'ticker', col]], on=['date', 'ticker'], how='left', validate='1:1')
            assert len(m) == len(ref) and m[col].notna().all(), 'coverage'
            return m[col].values.astype('float64')
        P = {n: cover(g, 'pred') for n, g in fc.groupby('config')}
        for n in ETAL:
            P[n] = cover(pd.read_parquet(Z.ROOT / 'data/regen/gl3_zh190/etalony' / f'{n}_h{h}.parquet'), 'prediction')
        M = {n: V23.metrics(ref, p, st) for n, p in P.items()}
        assert set(M) == set(tab.index), (h, set(M) ^ set(tab.index))
        for name, (mine, tm) in M.items():
            for key, v in mine.items():
                worst, count = max(worst, abs(float(tab.loc[name, key]) - v)), count + 1
            for e in ETAL:
                assert int((tm < M[e][1] - 1e-12).sum()) == tab.loc[name, f'tickers_better_vs_{e}'], (h, name, e)
    return dict(max_abs_diff=worst, values_compared=count, ok=bool(worst <= 1e-11 and count > 0))


def g5(ev):
    sel, rows = json.loads((Z.TAB / 'selection.json').read_text()), {}
    for h in Z.HS:
        t = pd.read_csv(Z.TAB / f'rows_h{h}.csv')
        rows[str(h)] = len(t)
        assert set(NEED[h]) | set(ETAL) <= set(t.config) and 'cel' in t.columns and not t.cel.isna().any(), h
        assert {'final_before_seeds', 'picks', 'closed_window_ok'} <= set(sel[str(h)]) and sel[str(h)]['closed_window_ok'], h
    f = pd.DataFrame(ev)
    cnt = dict(fits_by_h={str(h): {ph: int(((f.h == h) & (f.phase == ph)).sum()) for ph in sorted(f.phase.unique())} for h in Z.HS},
               ledger_events=len(f), by_component=f.groupby('component').size().to_dict(), rows_by_h=rows,
               series_by_h={str(h): int(pd.read_parquet(Z.OUT / f'forecasts_h{h}.parquet').config.nunique()) for h in Z.HS},
               cells={str(h): len(list((Z.OUT / 'cells').glob(f'h{h}_*_s*.parquet'))) // 1 for h in Z.HS}, seeds=list(Z.SEEDS))
    (Z.TAB / 'attempt_counts.json').write_text(json.dumps(cnt, indent=1, default=int))
    return dict(rows=rows, ok=True)


def verify():
    ev, r3 = g3()
    out = dict(G2=g2(), G3=r3, G4=g4(), G5=g5(ev))
    (Z.TAB / 'verify.json').write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'series'} for k, v in out.items()}, indent=1, default=str))
    assert all(v['ok'] for v in out.values())


if __name__ == '__main__':
    verify()
