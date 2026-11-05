# -*- coding: utf-8 -*-
"""Zh-190 stage 1 pipeline: matrices -> screening (20 x 3) -> top 7 on five years -> Optuna by groups -> 10 seeds -> B*.

Usage: PYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_zh190_boost.py <huber|matrix|g2|screen|five|iter3|iter4|bstar|all>
"""
import json
import sys
import time
import numpy as np
import pandas as pd
import gl3_zh190_boost_core as K
import gl3_zh190_boost_eval as E
import gl3_zh190_boost_jobs as J

SCREEN_YEARS, ALL_YEARS = (2019, 2020), tuple(range(2016, 2022))
TOP_N = 7


def log(msg):
    with (K.TAB / 'run.log').open('a') as f:
        f.write(f'{pd.Timestamp.now(tz="UTC").isoformat()} {msg}\n')


def huber():
    t = {f'huber{m}': K.HUBER_ALPHA * m for m in (1, 2, 4)}
    (K.TAB / 'huber_threshold.json').write_text(json.dumps(dict(
        written_before_counting=True, lightgbm_alpha_default=K.HUBER_ALPHA, thresholds=t,
        source='l2_lib.py:651 (LightGBM objective huber, no alpha) and fin5params.lgbm (no alpha key): the final uses the LightGBM default',
        qlike='B.qlike_grad (K10): grad = 1 - exp(y-f), hess = exp(y-f), no exp(s2/2) correction'), indent=1))


def matrix(d, eng):
    b = d.frames['original']
    for h in K.HS:
        x, names, _ = eng.static(h)
        m = pd.DataFrame(x, columns=names)
        m.insert(0, 'date', b.date.values)
        m.insert(1, 'ticker', b.ticker.values)
        m['sector'] = b.ticker.map(K.SECTORS).values
        m['target'] = d.y(h, 'original').astype('float32')
        m[f'_end{h}'] = b[f'_end{h}'].values
        assert m.date.max() <= pd.Timestamp('2021-12-31')
        m.to_parquet(K.OUT / f'matrix_h{h}.parquet', index=False)
        log(f'matrix h{h} {m.shape}')


def cfgs(rows):
    return [(r.scheme, r.loss, r.optuna, r.bag) for r in rows.itertuples()]


def table(sc, etal, h, stage, items, years=ALL_YEARS):
    out = []
    for scheme, loss, opt, bag in items:
        c = J.cid(scheme, loss, opt, bag)
        df = E.read_cells(h, c, years)
        out.append(dict(h=h, stage=stage, config=c, scheme=scheme, loss=loss, optuna=opt, bag=bag, **sc.row(sc.align(df), etal)))
    return pd.DataFrame(out).sort_values('mean_QLIKE', kind='stable').reset_index(drop=True)


def screen(d):
    jobs = [('base', h, y, s, list(K.LOSSES)) for h in K.HS for y in SCREEN_YEARS for s in K.SCHEMES]
    log(f'screen {len(jobs)} jobs, {J.run_pool(jobs, K.TAB / "run.log"):.0f}s')
    for h in K.HS:
        sc, rows = E.Scorer(d, h), []
        for s in K.SCHEMES:
            for l in K.LOSSES:
                ym = sc.year_means(E.read_cells(h, J.cid(s, l), SCREEN_YEARS), SCREEN_YEARS)
                rows.append(dict(h=h, scheme=s, loss=l, QLIKE_2019=ym[2019], QLIKE_2020=ym[2020], mean_2019_2020=(ym[2019] + ym[2020]) / 2))
        t = pd.DataFrame(rows).sort_values(['mean_2019_2020', 'scheme', 'loss']).reset_index(drop=True)
        t['rank'], t['passed'] = np.arange(1, len(t) + 1), np.arange(len(t)) < TOP_N
        t.to_csv(K.TAB / f'screen_h{h}.csv', index=False)


def passed(h):
    t = pd.read_csv(K.TAB / f'screen_h{h}.csv')
    return t[t.passed].assign(optuna=False, bag=False)


def five(d):
    jobs = []
    for h in K.HS:
        p = passed(h)
        jobs += [('base', h, y, s, list(p[p.scheme == s].loss)) for y in ALL_YEARS if y not in SCREEN_YEARS for s in K.SCHEMES if (p.scheme == s).any()]
    log(f'five {len(jobs)} jobs, {J.run_pool(jobs, K.TAB / "run.log"):.0f}s')
    for h in K.HS:
        sc = E.Scorer(d, h)
        table(sc, sc.etalons(), h, 'stage1', cfgs(passed(h))).to_csv(K.TAB / f'stage1_h{h}.csv', index=False)


def read_table(name, h):
    return pd.read_csv(K.TAB / f'{name}_h{h}.csv')


def iter3(d):
    jobs = []
    for h in K.HS:
        top = read_table('stage1', h).head(2)
        jobs += [('opt', h, y, r.scheme, [r.loss]) for r in top.itertuples() for y in ALL_YEARS]
    log(f'iter3 {len(jobs)} jobs, {J.run_pool(jobs, K.TAB / "run.log"):.0f}s')
    for h in K.HS:
        sc = E.Scorer(d, h)
        top = read_table('stage1', h).head(2)
        table(sc, sc.etalons(), h, 'iter3', [(r.scheme, r.loss, True, False) for r in top.itertuples()]).to_csv(K.TAB / f'iter3_h{h}.csv', index=False)
        rows = [r for p in sorted((K.OUT / 'optuna').glob(f'h{h}_*.json')) for r in json.loads(p.read_text())['trials']]
        pd.DataFrame(rows).assign(h=h, params=lambda t: t.params.map(json.dumps)).to_csv(K.TAB / f'optuna_trials_h{h}.csv', index=False)


def best_row(h):
    t = pd.concat([read_table('stage1', h), read_table('iter3', h)], ignore_index=True)
    return t.sort_values('mean_QLIKE', kind='stable').iloc[0]


def iter4(d):
    jobs = []
    for h in K.HS:
        r = best_row(h)
        jobs += [('bagopt' if r.optuna else 'bag', h, y, r.scheme, [r.loss]) for y in ALL_YEARS]
    log(f'iter4 {len(jobs)} jobs, {J.run_pool(jobs, K.TAB / "run.log"):.0f}s')
    for h in K.HS:
        sc, r = E.Scorer(d, h), best_row(h)
        table(sc, sc.etalons(), h, 'iter4', [(r.scheme, r.loss, bool(r.optuna), True)]).to_csv(K.TAB / f'iter4_h{h}.csv', index=False)


def bstar(d):
    """P7: the row with the smallest mean QLIKE among P1, P5, P6; forecasts with the raw log forecast and the correction; summary table."""
    allrows, star = [], []
    for h in K.HS:
        sc = E.Scorer(d, h)
        et = sc.etalons()
        t = pd.concat([read_table(n, h) for n in ('stage1', 'iter3', 'iter4')], ignore_index=True)
        ref = [dict(h=h, stage='etalon', config=n, **{k: v for k, v in sc.row(p, {}).items()}) for n, p in et.items()]
        t = pd.concat([pd.DataFrame(ref), t], ignore_index=True)
        w = t[t.stage != 'etalon'].sort_values('mean_QLIKE', kind='stable').iloc[0]
        t['bstar'] = (t.stage == w.stage) & (t.config == w.config)
        allrows.append(t)
        cells = E.read_cells(h, w.config)
        cells.to_parquet(K.OUT / f'bstar_h{h}.parquet', index=False)
        star.append(dict(h=h, stage=w.stage, config=w.config, mean_QLIKE=w.mean_QLIKE))
        fc = [E.read_cells(h, J.cid(r.scheme, r.loss, r.optuna, r.bag)).assign(stage=r.stage, config=r.config) for r in t[t.stage != 'etalon'].itertuples()]
        pd.concat(fc, ignore_index=True)[['stage', 'config', 'date', 'ticker', 'pred']].to_parquet(K.OUT / f'forecasts_h{h}.parquet', index=False)
    pd.concat(allrows, ignore_index=True).to_csv(K.TAB / 'stage1_summary.csv', index=False)
    pd.DataFrame(star).to_csv(K.TAB / 'bstar.csv', index=False)


def main(stage):
    d, eng = K.setup()
    J.G.update(d=d, eng=eng)
    t0 = time.time()
    steps = dict(huber=lambda: huber(), matrix=lambda: matrix(d, eng), screen=lambda: screen(d), five=lambda: five(d),
                 iter3=lambda: iter3(d), iter4=lambda: iter4(d), bstar=lambda: bstar(d))
    for name in (list(steps) if stage == 'all' else [stage]):
        steps[name]()
        log(f'stage {name} done {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main(sys.argv[1])
