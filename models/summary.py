"""Сводка валидации: одна таблица по h и наборам (разработочный val 2016-2021 = шаги 1-6; подтверждающий val 2022-2024 = шаги 7-9,
только если прогон был с --confirm/--finalists). Тестовые метрики — только --scope test и только при docs/otchety/FINAL_DESIGN.md."""
import numpy as np, pandas as pd
from . import cache as K, config as C, hybrid as HY


def require_final():
    if not C.FINAL.exists():
        raise SystemExit('--scope test закрыт: нет docs/otchety/FINAL_DESIGN.md (тестовые прогнозы лежат в data/models_cache/test_locked/)')


def read_test(kind, k):
    require_final()
    return pd.read_parquet(K.pred_path('test', kind, k))


def _q(d):
    d = d[d.rv_actual > 1e-12]  # фильтр gibrid_io.merged (v3.py:65)
    return d.groupby('step').apply(lambda g: C.qlike(g.rv_actual.values, g.rv_pred.values))


def hyb_tables(man, h):
    d = K.stage_dir('hybrid', man['hybrid'][str(h)])
    return pd.read_csv(d / 'designs.csv'), pd.read_csv(d / 'final.csv'), pd.read_csv(d / 'single.csv')


def per_step(man, scope='val'):
    """Длинная таблица (h, модель, step, q). val — из таблиц стадии гибрида; test — из test_locked (после FINAL_DESIGN)."""
    rows = []
    sp = pd.read_csv(K.stage_dir('har_spec', man['har_spec']) / 'spec.csv') if scope == 'val' else None
    for h in map(int, man['hybrid']):
        R, win, single = hyb_tables(man, h)
        fin = win.design.iloc[0]
        if scope == 'val':
            b = R[R.design == 'single|step|B0.1oos'].set_index('step')
            ser = {'HAR-J (а)': single.set_index('step').q_har_raw, 'HAR-J (б) Б0.1oos': b.q_har,
                   'HAR-J (в) окно этапа 2б': sp[sp.h == h].set_index('step').q_chosen,
                   'XGBoost': single.set_index('step').q_xgb_raw, 'LightGBM': single.set_index('step').q_lgb_raw,
                   f'Гибрид как в коде ({HY.CODE_DESIGN})': R[R.design == HY.CODE_DESIGN].set_index('step').q_hyb,
                   f'Гибрид по правилу ({fin})': R[R.design == fin].set_index('step').q_hyb}
        else:
            hk = man['hybrid'][str(h)]
            ser = {'HAR-J (а)': _q(read_test('har', man['har'][str(h)])),
                   'XGBoost': _q(pd.concat([read_test('boost', k) for k in man['boost'][f'xgb_{h}']])),
                   'LightGBM': _q(pd.concat([read_test('boost', k) for k in man['boost'][f'lgbm_{h}']])),
                   f'Гибрид как в коде ({HY.CODE_DESIGN})': _q(read_test('hyb', f'{hk}_{HY.fname(HY.CODE_DESIGN)}')),
                   f'Гибрид по правилу ({fin})': _q(read_test('hyb', f'{hk}_{HY.fname(fin)}'))}
        for name, s in ser.items():
            for st, q in s.items():
                rows.append(dict(h=h, model=name, step=int(st), q=float(q)))
    return pd.DataFrame(rows)


def table(man, scope='val', confirm=False):
    L = per_step(man, scope).sort_values('h', kind='stable').reset_index(drop=True)
    if scope == 'val' and not confirm:
        L = L[L.step <= C.DEV]
    sets = {'разраб. 2016-2021' if scope == 'val' else 'тест 2017-2022': L.step <= C.DEV}
    if confirm or scope == 'test':
        sets['подтв. 2022-2024' if scope == 'val' else 'тест 2023-2025'] = L.step > C.DEV
    out = []
    for (h, m), g in L.groupby(['h', 'model'], sort=False):
        r = {'h': h, 'модель': m}
        for nm, mask in sets.items():
            v = g[mask.loc[g.index]].q
            r[nm] = round(float(v.mean()), 4) if len(v) else np.nan
        out.append(r)
    return pd.DataFrame(out)
