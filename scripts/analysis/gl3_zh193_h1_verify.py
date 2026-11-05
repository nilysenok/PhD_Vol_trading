# -*- coding: utf-8 -*-
"""Zh-193 gates G2 (ledger), G3 (independent recomputation of the metrics from forecasts_h1.parquet, own numpy code), G4 (rows, selection.json,
attempt_counts.json) and the reproduction check against stage 1. Usage: PYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_zh193_h1_verify.py"""
import json
import numpy as np
import pandas as pd
import gl3_zh193_h1_fit as Z
from gl3_zh190_boost_checks import rvi_stress, REF, LAST
from gl3_zh190_stage23_verify import metrics

ETAL = {'FIN_PIT': 'FIN_PIT', 'HAR-J(b)': 'HARJb', 'HAR_i': 'HARi_LOYO'}
NEED = ['Bstar_h1', 'B1', 'B2_mono', 'B2_lt', 'B2_both', 'B3_qlike', 'B3_huber8', 'B4_hl1', 'B4_hl2', 'B5_d4', 'B5_d6', 'B6', 'Bss',
        'G1_c50', 'G1_c75', 'G1_c100', 'G1', 'G2', 'G3', 'G4_hard', 'G4_logit', 'G4', 'G5', 'G6', 'G7']


def g2():
    ev = [json.loads(x) for x in Z.EVENTS.read_text().splitlines()]
    bad = [e for e in ev if e['h'] != 1 or not (pd.Timestamp(e['max_target_end']) < pd.Timestamp(e['cutoff']) < pd.Timestamp('2022-01-01'))]
    return ev, dict(events=len(ev), bad=len(bad), horizons=sorted({e['h'] for e in ev}), ok=bool(ev and not bad and {e['h'] for e in ev} == {1}))


def g3():
    ref = pd.read_parquet(str(REF).format(h=1), columns=['date', 'ticker', 'year', 'rv_actual'], filters=[('date', '<=', LAST)])
    ref = ref[ref.date >= '2017-01-01'].reset_index(drop=True)
    tab = pd.read_csv(Z.TAB / 'summary.csv').set_index('config')
    fc = pd.read_parquet(Z.OUT / 'forecasts_h1.parquet')
    assert fc.pred.dtype == np.float32 and fc.date.max() < pd.Timestamp('2022-01-01') and np.isfinite(fc.pred).all() and (fc.pred > 0).all()

    def cover(df, col):
        m = ref[['date', 'ticker']].merge(df[['date', 'ticker', col]], on=['date', 'ticker'], how='left', validate='1:1')
        assert len(m) == len(ref) and m[col].notna().all(), 'coverage'
        return m[col].values.astype('float64')
    P = {n: cover(g, 'pred') for n, g in fc.groupby('config')}
    for name, f in ETAL.items():
        P[name] = cover(pd.read_parquet(Z.K.OUT / 'etalony' / f'{f}_h1.parquet'), 'prediction')
    st, M, worst, count = rvi_stress(), {}, 0., 0
    M = {n: metrics(ref, p, st) for n, p in P.items()}
    for name, (mine, tm) in M.items():
        for key, v in mine.items():
            worst, count = max(worst, abs(float(tab.loc[name, key]) - v)), count + 1
        for e in ETAL:
            assert int((tm < M[e][1] - 1e-12).sum()) == tab.loc[name, f'tickers_better_vs_{ETAL[e]}'], (name, e)
    return dict(max_abs_diff=worst, values_compared=count, series=len(P), ok=bool(worst <= 1e-11 and count > 0))


def reproduction():
    """Own cat-huber1 cells (2019, 2020) against the stage-1 cells of the same configuration, and B* against the saved stage-1 forecast."""
    worst = 0.
    for Y in (2019, 2020):
        a, r = pd.read_parquet(Z.cell_path('S1_cat_huber1', Y)), pd.read_parquet(Z.K.OUT / 'cells' / f'h1_{Y}_cat-huber1.parquet')
        m = a.merge(r, on=['date', 'ticker'], suffixes=('', '_r'), validate='1:1')
        assert len(m) == len(a)
        worst = max(worst, float(np.abs(np.log(m.pred / m.pred_r)).max()))
    fc = pd.read_parquet(Z.OUT / 'forecasts_h1.parquet')
    b = pd.read_parquet(Z.K.OUT / 'bstar_h1.parquet')
    m = fc[fc.config == 'Bstar_h1'].merge(b, on=['date', 'ticker'], suffixes=('', '_r'), validate='1:1')
    return dict(max_abs_dlog_cat=worst, bstar_rows=len(m), max_abs_dlog_bstar=float(np.abs(np.log(m.pred / m.pred_r)).max()), ok=bool(worst < 1e-6))


def g4(ev):
    t, sel = pd.read_csv(Z.TAB / 'summary.csv'), json.loads((Z.TAB / 'selection.json').read_text())
    miss = [n for n in NEED + list(ETAL) if n not in set(t.config)]
    assert not miss, miss
    assert {'Bss', 'G1_pick', 'G4_pick', 'G5_pick', 'base_G5', 'base_G6', 'G7_of', 'Bcomb_items'} <= set(sel) and 'cel' in t.columns
    f = pd.DataFrame(ev)
    cnt = dict(rows=len(t), series=int(pd.read_parquet(Z.OUT / 'forecasts_h1.parquet').config.nunique()), ledger_events=len(f),
               by_component=f.groupby('component').size().to_dict(), by_phase=f.groupby('phase').size().to_dict(),
               cells=len(list((Z.OUT / 'cells').glob('*.parquet'))), cutoffs=sorted(f.cutoff.str[:10].unique().tolist()))
    (Z.TAB / 'attempt_counts.json').write_text(json.dumps(cnt, indent=1, default=int))
    return dict(rows=len(t), missing=miss, ok=True)


def verify():
    ev, r2 = g2()
    out = dict(G2=r2, G3=g3(), reproduction=reproduction(), G4=g4(ev))
    (Z.TAB / 'verify.json').write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))
    assert all(v['ok'] for v in out.values())


if __name__ == '__main__':
    verify()
