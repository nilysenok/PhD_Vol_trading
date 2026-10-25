"""Сводка валидации: одна таблица по h и наборам (разработочный val 2017-2021 = шаги 2-6 (метрика гибрида вне выборки, gibrid_v17 --dev-eval oos); подтверждающий val 2022-2024 = шаги 7-9,
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
    return (pd.read_csv(d / f'designs_val_P1_h{h}.csv'), pd.read_csv(d / f'final_choice_P1_h{h}.csv'), pd.read_csv(d / 'single.csv'),
            pd.read_csv(d / f'har_specs_val_P1_h{h}.csv'))


def lo(man):
    """Первый шаг разработочного набора: метрика гибрида вне выборки (gibrid_v17 --dev-eval oos) -> шаги 2-6."""
    return int(K.meta('hybrid', next(iter(man['hybrid'].values()))).get('lo', 1))


def per_step(man, scope='val'):
    """Длинная таблица (h, модель, step, q). val — из таблиц стадии гибрида; test — из test_locked (после FINAL_DESIGN)."""
    rows = []
    for h in map(int, man['hybrid']):
        R, win, single, sp = hyb_tables(man, h)
        fin = win.design.iloc[0]
        if scope == 'val':
            b = R[R.design == 'single|step|B-ord|L0'].set_index('step')
            ser = {'HAR-J (а)': single.set_index('step').q_har_raw, 'HAR-J (б) Б0.1 ord': b.q_har,
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
    l0 = lo(man) if scope == 'val' else 1
    L = L[L.step >= l0]
    if scope == 'val' and not confirm:
        L = L[L.step <= C.DEV]
    sets = {f'разраб. {2015 + l0}-2021' if scope == 'val' else 'тест 2017-2022': L.step <= C.DEV}
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


def compare(base, others):
    """Правило рычагов 25.10: остаётся при улучшении QLIKE разработочного набора >= 1% к базе или снятии нарушения критериев (по h).
    base/others — манифесты dev-прогонов. Строки: одиночные XGBoost/LightGBM и гибрид по правилу."""
    def pick(man):
        t = table(man)
        t = t.set_index(['h', 'модель'])[t.columns[2]]
        fin = {h: hyb_tables(man, h)[1].iloc[0] for h in map(int, man['hybrid'])}
        return t, fin
    tb, fb = pick(base)
    rows = []
    for man in others:
        to, fo = pick(man)
        for h in sorted(fo):
            rb = {m.split(' (')[0]: v for (hh, m), v in tb.items() if hh == h}
            ro = {m.split(' (')[0]: v for (hh, m), v in to.items() if hh == h}
            gain = {m: (rb[m] - ro[m]) / rb[m] for m in ('XGBoost', 'LightGBM', 'Гибрид по правилу') if m in rb and m in ro}
            keep = gain.get('Гибрид по правилу', 0) >= 0.01 or fo[h].n_failed < fb[h].n_failed
            rows.append({'h': h, 'дизайн': man['design'], **{f'Δ% {m}': round(100 * g, 2) for m, g in gain.items()},
                         'нарушено база': fb[h].failed if isinstance(fb[h].failed, str) else '-', 'нарушено': fo[h].failed if isinstance(fo[h].failed, str) else '-',
                         'гибрид': fo[h].design, 'остаётся': 'да' if keep else 'нет'})
    return pd.DataFrame(rows)
