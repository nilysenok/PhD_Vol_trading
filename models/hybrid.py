"""HAR-J-семейство и гибрид (воркеры). Логика не переписана: HAR-J — gibrid_har.har_forecasts (walk_forward_full.py:164-205),
строка (в) — gibrid_har_spec.run (окна этапа 2б, etap2_okno.py), гибрид V1-V7 + V0, дизайны single x (step|minimax) x (Б0.1oos|без),
критерии c1-c4 и правило выбора 25.10 — функции gibrid_v17 / gibrid_designs (цикл main() gibrid_v17.py для варианта single, по одному h).
Шаги ограничиваются фильтром данных (год <= тестового года последнего шага): расширяющиеся окна не зависят от будущих лет."""
import functools, time
import warnings
import numpy as np, pandas as pd
from . import cache as K, config as C, data as D

warnings.filterwarnings('ignore')  # как tune_B.py / walk_forward_full.py; логи — в data/run/

HYB_CODE = [C.ANALYSIS / f for f in ('gibrid_v17.py', 'gibrid_schemes.py', 'gibrid_designs.py', 'gibrid_io.py')]
CR = ('c1', 'c2', 'c3', 'c4')
CODE_DESIGN = 'single|step|bez'  # «как в коде» (hyb_*_h{h}_*.parquet gibrid_v17.py)


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


def hyb_key(h, hk, bkeys, cfg):
    return K.key('hybrid', h=h, har=hk, boost=bkeys, cfg=cfg, code=C.code_hash('hybrid.py', *HYB_CODE))


def _boost(keys):
    return {s: pd.concat([pd.read_parquet(K.pred_path(s, 'boost', k)) for k in keys], ignore_index=True) for s in ('val', 'test')}


def fname(design):
    return design.replace('|', '__')


@_timed
def run_hybrid(t):
    """t: dict(h, har, boost={'xgb': [ключи по шагам], 'lgbm': [...]}, cfg={'v7_lag': 'h'|1, 'ref': bool})."""
    import gibrid_v17 as G
    import gibrid_designs as DS
    import gibrid_schemes as S
    from gibrid_io import merged
    h, cfg = t['h'], t['cfg']
    k = hyb_key(h, t['har'], t['boost'], cfg)
    if K.done('hybrid', k):
        return dict(kind='hybrid', h=h, key=k, hit=True)
    hv, ht = (pd.read_parquet(K.pred_path(s, 'har', t['har'])) for s in ('val', 'test'))
    bx, bl = _boost(t['boost']['xgb']), _boost(t['boost']['lgbm'])
    lag = h if str(cfg.get('v7_lag', 'h')) == 'h' else int(cfg['v7_lag'])
    rows, outs, blog, ref = [], {}, [], []

    def add(ch, ov, ot, name):  # = add() в gibrid_v17.main
        ch.insert(0, 'design', name)
        fl = DS.flags(ch[ch.step <= C.DEV])
        ch['ratio_har'] = ch.q_hyb / ch.q_har
        for c in CR + ('worst_ratio', 'worst_ratio_single', 'max_w_jump', 'ml_switches'):
            ch[c + '_h'] = fl[c]
        rows.append(ch); outs[name] = (ov, ot)

    etalon = None
    for bias in (2, 0):
        lg = []
        res = G.compute(hv, ht, h, bx, bl, bias, lag, lg)
        if bias == 2:
            etalon = [{c: r['base'][c] for c in ('q_har', 'q_xgb_single', 'q_lgb_single')} for r in res]
            blog += lg
        else:
            for r, e in zip(res, etalon):
                r['base'].update(e)
        for sel in ('step', 'minimax'):
            add(*DS.select(res, sel, h), f'single|{sel}|{"B0.1oos" if bias else "bez"}')
        if bias == 2:
            for comp in ('har', 'xgb', 'lgb'):
                add(*DS.v0(res, comp, h), f'single|V0_{comp.upper()}|B0.1oos')
    if cfg.get('ref', True):  # справка gibrid_v17: Б0.1 in-sample и V7 lag=1
        c1, c0, cl = (DS.select(G.compute(hv, ht, h, bx, bl, b, l, []), 'step', h)[0] for b, l in ((1, lag), (0, lag), (0, 1)))
        m = lambda c, col: c.loc[c.step <= C.DEV, col].mean()
        ref.append(dict(h=h, hyb_B01_insample=m(c1, 'q_hyb'), har_B01_insample=m(c1, 'q_har'), hyb_bez_lag_h=m(c0, 'q_hyb'),
                        V7_lag_h=m(c0, 'q_V7_Rolling_W20'), V7_lag1_orig=m(cl, 'q_V7_Rolling_W20'), hyb_bez_lag1=m(cl, 'q_hyb')))
    R = pd.concat(rows, ignore_index=True)
    Rd = R[R.step <= C.DEV]
    F = Rd.groupby(['h', 'design'], sort=False).agg(**{c: (c + '_h', 'first') for c in CR}, mean_q=('q_hyb', 'mean')).reset_index()
    F['failed'] = F.apply(lambda r: ','.join(c for c in CR if not r[c]), axis=1)
    F['n_failed'] = F.apply(lambda r: sum(not r[c] for c in CR), axis=1)
    win = F.sort_values(['h', 'n_failed', 'mean_q']).head(1)
    vm = merged(hv, bx['val'], bl['val'])  # одиночные без поправки на тех же строках, что гибрид
    single = pd.DataFrame([dict(h=h, step=s, val_year=int(g.year.iloc[0]), q_har_raw=S.qlike(g.rv_actual.values, g.har.values),
                                q_xgb_raw=S.qlike(g.rv_actual.values, g.xgb.values), q_lgb_raw=S.qlike(g.rv_actual.values, g.lgb.values))
                           for s, g in vm.groupby('step')])
    for name, (ov, ot) in outs.items():
        K.write_atomic(ov, K.pred_path('val', 'hyb', f'{k}_{fname(name)}'))
        K.write_atomic(ot, K.pred_path('test', 'hyb', f'{k}_{fname(name)}'))
    d = K.stage_dir('hybrid', k); d.mkdir(parents=True, exist_ok=True)
    R.to_csv(d / 'designs.csv', index=False); win.to_csv(d / 'final.csv', index=False); single.to_csv(d / 'single.csv', index=False)
    pd.DataFrame(blog).to_csv(d / 'bias_factors.csv', index=False); pd.DataFrame(ref).to_csv(d / 'ref.csv', index=False)
    K.mark('hybrid', k, h=h, final=win.design.iloc[0], code=CODE_DESIGN)
    return dict(kind='hybrid', h=h, key=k, hit=False)
