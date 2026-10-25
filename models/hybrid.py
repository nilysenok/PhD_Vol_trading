"""HAR-J-семейство и гибрид (воркеры). Логика не переписана: HAR-J — gibrid_har.har_forecasts (walk_forward_full.py:164-205),
строка (в) — gibrid_har_spec.run (окна этапа 2б, etap2_okno.py), гибрид — gibrid_v17.run_h по одному h (пространство дизайнов Д8: схемы,
поправки, усадка, смешивание rv/log, V0, критерии c1-c4 и правило выбора — без изменений; метрика разработочного набора — oos, шаги 2-6).
Шаги ограничиваются фильтром данных (год <= тестового года последнего шага): расширяющиеся окна не зависят от будущих лет."""
import functools, os, time
import warnings
import numpy as np, pandas as pd
from . import cache as K, config as C, data as D

warnings.filterwarnings('ignore')  # как tune_B.py / walk_forward_full.py; логи — в data/run/

HYB_CODE = [C.ANALYSIS / f for f in ('gibrid_v17.py', 'gibrid_schemes.py', 'gibrid_designs.py', 'gibrid_io.py', 'gibrid_compute.py', 'gibrid_bias.py',
                                      'gibrid_d8.py', 'gibrid_harcj.py', 'gibrid_har.py', 'gibrid_har_spec.py', 'gibrid_report.py', 'gibrid_v2sum.py')]
CR = ('c1', 'c2', 'c3', 'c4')
CODE_DESIGN = 'single|step|bez|L0'  # «как в коде» (hyb_*_h{h}_*.parquet gibrid_v17.py)


def _timed(f):
    @functools.wraps(f)
    def w(t):
        t0 = time.time()
        r = f(t)
        r['sec'] = time.time() - t0
        return r
    return w


def _cut(a, nsteps):
    return a[a.year <= C.YEARS[nsteps - 1]]


def har_key(dk, h, nsteps):
    return K.key('har', data=dk, h=h, nsteps=nsteps, code=C.code_hash('hybrid.py', C.ANALYSIS / 'gibrid_har.py'))


@_timed
def run_har(t):
    import gibrid_har as GH
    k = har_key(t['dk'], t['h'], t['nsteps'])
    if K.pred_path('val', 'har', k).exists() and K.pred_path('test', 'har', k).exists():
        return dict(kind='har', h=t['h'], key=k, hit=True)
    a, harj = D.read_all(t['dk'])
    hv, ht, _ = GH.har_forecasts(t['h'], _cut(a, t['nsteps']), harj)
    K.write_atomic(hv, K.pred_path('val', 'har', k)); K.write_atomic(ht, K.pred_path('test', 'har', k))
    return dict(kind='har', h=t['h'], key=k, hit=False)


def spec_key(dk, nsteps):
    return K.key('har_spec', data=dk, nsteps=nsteps, code=C.code_hash('hybrid.py', C.ANALYSIS / 'gibrid_har_spec.py', C.ANALYSIS / 'gibrid_har.py'))


@_timed
def run_spec(t):
    import gibrid_har_spec as HS
    k = spec_key(t['dk'], t['nsteps'])
    if K.done('har_spec', k):
        return dict(kind='har_spec', key=k, hit=True)
    a, harj = D.read_all(t['dk'])
    sp = HS.run(_cut(a, t['nsteps']), harj)  # только валидация
    d = K.stage_dir('har_spec', k); d.mkdir(parents=True, exist_ok=True)
    sp.to_csv(d / 'spec.csv', index=False)
    K.mark('har_spec', k)
    return dict(kind='har_spec', key=k, hit=False)


def hyb_key(h, hk, bkeys, cfg, dk=None, nsteps=None):
    return K.key('hybrid', h=h, har=hk, boost=bkeys, cfg=cfg, dk=dk, nsteps=nsteps, code=C.code_hash('hybrid.py', *HYB_CODE))


def _boost(keys):
    return {s: pd.concat([pd.read_parquet(K.pred_path(s, 'boost', k)) for k in keys], ignore_index=True) for s in ('val', 'test')}


def fname(design):
    return design.replace('|', '__')


@_timed
def run_hybrid(t):
    """t: dict(h, dk, nsteps, har, boost={'xgb': [ключи по шагам], 'lgbm': [...]}, cfg). Вызов gibrid_v17.run_h как есть: прогнозы конвейера
    кладутся во временный regen под именами П1 (xgb_h*, P1_h* — у П1 только вариант single), выходы run_h — в каталог стадии;
    test-прогнозы гибрида переносятся в test_locked/hyb."""
    import shutil
    from types import SimpleNamespace
    import gibrid_v17 as G
    import gibrid_harcj as HC
    import gibrid_schemes as S
    from gibrid_io import merged
    h, cfg = t['h'], t['cfg']
    k = hyb_key(h, t['har'], t['boost'], cfg, t['dk'], t['nsteps'])
    if K.done('hybrid', k):
        return dict(kind='hybrid', h=h, key=k, hit=True)
    d = K.stage_dir('hybrid', k); reg = d / 'regen'
    reg.mkdir(parents=True, exist_ok=True)
    B = {m: _boost(t['boost'][m]) for m in C.MODELS}
    for m, nm in (('xgb', 'xgb'), ('lgbm', 'P1')):
        for s in ('val', 'test'):
            B[m][s].to_parquet(reg / f'{nm}_h{h}_{s}.parquet', index=False)
    a, harj = D.read_all(t['dk'])
    A = HC.add_cj(_cut(a, t['nsteps']))
    args = SimpleNamespace(boost='P1', v7_lag=str(cfg.get('v7_lag', 'h')), blend=list(cfg.get('blend', ['rv', 'log'])),
                           dev_eval=cfg.get('dev_eval', 'oos'), etalon=cfg.get('etalon', 'same'), confirm=False)
    G.LO = 2 if args.dev_eval == 'oos' else 1
    p = G.run_h(h, A, harj, args, d, reg)
    shutil.rmtree(reg)
    fin = p['win'].design.iloc[0]
    for des, base in ((CODE_DESIGN, f'hyb_P1_h{h}'), (fin, f'hyb_P1_final_h{h}')):
        K.write_atomic(pd.read_parquet(d / f'{base}_val.parquet'), K.pred_path('val', 'hyb', f'{k}_{fname(des)}'))
        os.replace(d / f'{base}_test.parquet', K.pred_path('test', 'hyb', f'{k}_{fname(des)}'))
    hv, _, _ = __import__('gibrid_har').har_forecasts(h, _cut(a, t['nsteps']), harj)
    vm = merged(hv, B['xgb']['val'], B['lgbm']['val'])  # одиночные без поправки на тех же строках, что гибрид
    pd.DataFrame([dict(h=h, step=s, val_year=int(g.year.iloc[0]), q_har_raw=S.qlike(g.rv_actual.values, g.har.values),
                       q_xgb_raw=S.qlike(g.rv_actual.values, g.xgb.values), q_lgb_raw=S.qlike(g.rv_actual.values, g.lgb.values))
                  for s, g in vm.groupby('step')]).to_csv(d / 'single.csv', index=False)
    K.mark('hybrid', k, h=h, final=fin, code=CODE_DESIGN, lo=G.LO)
    return dict(kind='hybrid', h=h, key=k, hit=False)
