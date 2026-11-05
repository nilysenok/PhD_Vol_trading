# -*- coding: utf-8 -*-
"""Zh-190 stage 1 gates: G2 (reuse of K9/K10), G3 (ledger), G4 (independent metrics from the saved parquet), G5 (rows, attempt counts).

Usage: PYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_zh190_boost_checks.py <g2|verify>
"""
import json
import sys
import numpy as np
import pandas as pd
import gl3_r56_engine as B
import gl3_zh190_boost_core as K

REF = K.ROOT / 'data/regen/gl3_r3/dev/predictions_h{h}.parquet'
LAST = pd.Timestamp('2021-12-31')


def g2():
    """(panel, mse) and (panel, qlike) at h=5, cut 2021-01-01, K9 recipe: new trainer in compat mode vs Engine.boost of K9/K10."""
    d, eng = K.setup()
    h = 5
    logy = np.log(np.clip(d.y(h, 'original'), 1e-12, None))
    p = dict(learning_rate=.03, num_leaves=31, min_data_in_leaf=100, lambda_l2=10., feature_fraction=.8, bagging_fraction=.8)
    out = dict(fin_pit_lgb_leg_saved=False, note='FIN_PIT.parquet is the mean of the LGB and XGB level forecasts; the LGB leg alone is not stored, so the K9/K10 panel recipe is the reference',
               huber1_not_in_k9_k10=True, mse=[], qlike=[])
    for year in range(2016, 2022):
        cut = pd.Timestamp(f'{year}-01-01')
        x, names, base = eng.features(h, 'annual', 'original', cut)
        eng.current_names = names
        _, _, tr, w = eng.context(h, 'annual', 'original', cut)
        c = K.Ctx(d.frames['original'], x, logy, base, h, cut, compat=True, tr=tr)
        for loss in ('mse', 'qlike'):
            if loss == 'mse':
                ref = base + B.Engine.boost(eng, h, 'annual', 'original', cut, 42, 'lgb', x, logy - base, tr, w, 'a', tag='G2_K9_')
            else:
                ref = base + B.Engine.boost(eng, h, 'annual', 'original', cut, 42, 'lgb', x, logy, tr, w, 'a', objective='qlike', base=base, c=1., tag='G2_K10_')
            L = K.learn(c, x, loss, p, np.ones(len(x), bool), ('g2', year), 42, (), rounds=300, patience=20, tag=f'G2_{loss}')
            rel = np.abs(base + L['full'].predict(x, raw_score=True) - ref)[c.pw] / np.abs(ref[c.pw])
            out[loss].append(dict(cutoff=str(cut.date()), reference='K9 (Engine.boost, lgb, style a)' if loss == 'mse' else 'K10 (Engine.boost, qlike, c=1)',
                                  max_rel_diff=float(rel.max()), rows=int(c.pw.sum()), trees=L['trees']))
    out['ok'] = all(r['max_rel_diff'] <= 1e-6 for k in ('mse', 'qlike') for r in out[k])
    (K.TAB / 'g2_reuse.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


def g3():
    ev = [json.loads(x) for x in (K.TAB / 'fit_events.jsonl').read_text().splitlines()]
    bad = [e for e in ev if e['h'] not in (1, 5, 22) or not (pd.Timestamp(e['max_target_end']) < pd.Timestamp(e['cutoff']) < pd.Timestamp('2022-01-01'))]
    return ev, dict(events=len(ev), bad=len(bad), horizons=sorted({e['h'] for e in ev}), ok=not bad)


def rvi_stress():
    a = pd.read_parquet(K.ROOT / 'data/regen/features_pit_v3.parquet', columns=['date', 'close_RVI'], filters=[('date', '<=', LAST)])
    s = a.groupby('date').close_RVI.first().sort_index()
    rank = s.rolling(253, min_periods=253).apply(lambda z: np.mean(z[:-1] <= z[-1]), raw=True)
    return (rank > 2 / 3)


def g4():
    """Independent recomputation (own numpy code, no gl3_r6b_common) of annual QLIKE, mean and stress rows from forecasts_h*.parquet."""
    st, res = rvi_stress(), {}
    tabs = pd.concat([pd.read_csv(K.TAB / f'{n}_h{h}.csv') for n in ('stage1', 'iter3', 'iter4') for h in K.HS], ignore_index=True)
    worst, n = 0., 0
    for h in K.HS:
        ref = pd.read_parquet(str(REF).format(h=h), columns=['date', 'ticker', 'year', 'rv_actual'], filters=[('date', '<=', LAST)])
        ref = ref[ref.date >= '2017-01-01'].reset_index(drop=True)
        fc = pd.read_parquet(K.OUT / f'forecasts_h{h}.parquet')
        assert fc.pred.dtype == np.float32 and fc.date.max() < pd.Timestamp('2022-01-01') and fc.date.min() >= pd.Timestamp('2016-01-01')
        for (stage, cfg), g in fc.groupby(['stage', 'config']):
            assert np.isfinite(g.pred).all() and (g.pred > 0).all()
            m = ref.merge(g[['date', 'ticker', 'pred']], on=['date', 'ticker'], how='left')
            assert len(m) == len(ref) and m.pred.notna().all(), 'coverage'
            r = m.rv_actual.values / m.pred.values.astype('float64')
            l = r - np.log(r) - 1
            ann = [l[m.year.values == y].mean() for y in range(2017, 2022)]
            s2020 = m.date.between('2020-03-01', '2020-06-30').values
            hi = m.date.map(st).astype(bool).values
            mine = dict(mean_QLIKE=np.mean(ann), stress2020=l[s2020].mean(), stressRVI=l[hi].mean(),
                        ratio_stress2020=r[s2020].mean(), ratio_stressRVI=r[hi].mean(), **{f'QLIKE_{y}': a for y, a in zip(range(2017, 2022), ann)})
            row = tabs[(tabs.h == h) & (tabs.stage == stage) & (tabs.config == cfg)]
            assert len(row) == 1, (h, stage, cfg)
            for k, v in mine.items():
                worst = max(worst, abs(float(row[k].iloc[0]) - v))
                n += 1
    return dict(max_abs_diff=worst, values_compared=n, ok=bool(worst <= 1e-11 and n > 0))


def g5(ev):
    cnt = lambda name: sum(len(pd.read_csv(K.TAB / f'{name}_h{h}.csv')) for h in K.HS)
    rows = dict(screen=cnt('screen'), stage1=cnt('stage1'), iter3=cnt('iter3'), iter4=cnt('iter4'), bstar=len(pd.read_csv(K.TAB / 'bstar.csv')))
    want = dict(screen=60, stage1=21, iter3=6, iter4=3, bstar=3)
    f = pd.DataFrame(ev)
    f['kind'] = f.component.str.split('_').str[:2].str.join('_')
    fits = {str(h): dict(refit=int(((f.h == h) & (f.phase == 'refit')).sum()), inner_ES=int(((f.h == h) & (f.phase == 'inner_ES')).sum())) for h in K.HS}
    trials = sum(len(pd.read_csv(K.TAB / f'optuna_trials_h{h}.csv')) for h in K.HS)
    cells = {str(h): len(list((K.OUT / 'cells').glob(f'h{h}_*.parquet'))) for h in K.HS}
    cfg = {str(h): int(sum(pd.read_csv(K.TAB / f'{n}_h{h}.csv').shape[0] for n in ('screen', 'stage1', 'iter3', 'iter4'))) for h in K.HS}
    (K.TAB / 'attempt_counts.json').write_text(json.dumps(dict(fits_by_h=fits, fit_events_total=len(f), fit_by_component=f.groupby('component').size().to_dict(),
                                                              optuna_trials=trials, cells_by_h=cells, config_rows_by_h=cfg,
                                                              configurations_screen_per_h=20, seeds_bagging=10), indent=1, default=int))
    return dict(rows=rows, want=want, ok=rows == want)


def matrices():
    """Matrix per h: float32 features (names of feature_names[h] with ln_ for rv_d/rv_w/rv_m), date <= 2021-12-31, target, _end, ticker, sector."""
    names = json.loads((K.ROOT / 'docs/tables/gl3_r5_r6/frozen_warmup_configuration.json').read_text())['feature_names']
    out = {}
    for h in K.HS:
        m = pd.read_parquet(K.OUT / f'matrix_h{h}.parquet')
        feat = [c for c in m.columns if c not in ('date', 'ticker', 'sector', 'target', f'_end{h}')]
        want = {('ln_' + n if n in ('rv_d', 'rv_w', 'rv_m') else n) for n in names[str(h)]}
        out[str(h)] = dict(rows=len(m), features=len(feat), same_names_as_json=set(feat) == want, all_float32=all(m[c].dtype == np.float32 for c in feat),
                           last_date=str(m.date.max().date()), no_model_outputs=not {'har_feature', 'ln_HAR_b'} & set(m.columns))
        out[str(h)]['ok'] = bool(out[str(h)]['same_names_as_json'] and out[str(h)]['all_float32'] and m.date.max() <= LAST and out[str(h)]['no_model_outputs'])
    return out


def verify():
    ev, r3 = g3()
    out = dict(G3=r3, G4=g4(), G5=g5(ev), matrices=matrices())
    (K.TAB / 'verify.json').write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == '__main__':
    {'g2': g2, 'verify': verify}[sys.argv[1]]()
