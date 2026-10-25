"""Таблицы главы 3 в CSV с фиксированными именами docs/tables/table_3_<N>_<набор>.csv (колонки — как в тексте диссертации
«Диссертация_Лысенок_правки_5.docx»: 3.2 «Год | QLIKE H = 1 | ...», 3.3 «Горизонт | QLIKE», 3.4/3.6/3.7 «Модель | H = 1 | H = 5 | H = 22»,
3.8 «H | t-стат. | p-value»; для 3.4/3.6/3.7 дополнительно _rmse/_mae). Набор: val_dev (val 2016-2021), val_all (+2022-2024, --confirm),
test (только после FINAL_DESIGN, --scope test). 3.1 (GJR-GARCH) и 3.5 (LSTM/GRU) конвейер не строит — этих моделей в нём нет.
HAR-J = строка (а) (расширяющееся окно, как в главе 3), бустинг — без поправки, «Гибридная» — дизайн по правилу 25.10."""
import json
import numpy as np, pandas as pd
from . import cache as K, config as C, hybrid as HY, pipeline as PL, summary as SM

MODEL_ORDER = ('HAR-J', 'XGBoost', 'LightGBM', 'Гибридная')
YEAR_ONE = 2019  # «тест 2019» в табл. 3.4 / 3.6


def preds(man, scope, h):
    rd = (lambda kind, k: pd.read_parquet(K.pred_path('val', kind, k))) if scope == 'val' else SM.read_test
    hk = man['hybrid'][str(h)]
    fin = K.meta('hybrid', hk)['final']
    P = {'HAR-J': rd('har', man['har'][str(h)]),
         'XGBoost': pd.concat([rd('boost', k) for k in man['boost'][f'xgb_{h}']]),
         'LightGBM': pd.concat([rd('boost', k) for k in man['boost'][f'lgbm_{h}']]),
         'Гибридная': rd('hyb', f'{hk}_{HY.fname(fin)}')}
    return {m: d[d.rv_actual > 1e-12] for m, d in P.items()}  # фильтр gibrid_io.merged


def by_year(d):
    out = {}
    for y, g in d.groupby('year'):
        e = g.rv_actual.values - g.rv_pred.values
        out[int(y)] = dict(QLIKE=float(C.qlike(g.rv_actual.values, g.rv_pred.values)), RMSE=float(np.sqrt(np.mean(e ** 2))), MAE=float(np.mean(np.abs(e))))
    return out


def fmt(v, metric):
    return round(v, 4) if metric == 'QLIKE' else float(f'{v:.4g}')


def run(design, scope, confirm, root):
    p = PL.manifest_path(design['name'], confirm and scope == 'val', False, root)
    if not p.exists():
        raise SystemExit(f'нет прогона {p.name}: сначала python3 -m models run --design ... {"--confirm" if confirm else ""}')
    man = json.loads(p.read_text())
    if scope == 'test':
        SM.require_final()
    tag = 'test' if scope == 'test' else ('val_all' if confirm else 'val_dev')
    M, dm = {}, []
    for h in C.HS:
        P = preds(man, scope, h)
        for m, d in P.items():
            M[(m, h)] = by_year(d)
        a = P['HAR-J'].merge(P['Гибридная'][['date', 'ticker', 'rv_pred']], on=['date', 'ticker'], suffixes=('_har', '_hyb')).sort_values(['date', 'ticker'])
        t, pv = C.dm_test(C.qlike_losses(a.rv_actual.values, a.rv_pred_har.values), C.qlike_losses(a.rv_actual.values, a.rv_pred_hyb.values), h)
        dm.append({'H': h, 't-стат.': round(float(t), 2), 'p-value': '<0,001' if pv < 1e-3 else round(float(pv), 3)})
    years = sorted(M[('HAR-J', 1)])
    span = f'{years[0]}-{years[-1]}'
    T = {'3_2': pd.DataFrame([{'Год': y, **{f'QLIKE H = {h}': fmt(M[('HAR-J', h)][y]['QLIKE'], 'QLIKE') for h in C.HS}} for y in years]
                             + [{'Год': f'Среднее {span}', **{f'QLIKE H = {h}': fmt(np.mean([M[('HAR-J', h)][y]['QLIKE'] for y in years]), 'QLIKE')
                                                            for h in C.HS}}]),
         '3_3': pd.DataFrame([{'Горизонт': f'H = {h}', 'QLIKE': fmt(np.mean([M[('HAR-J', h)][y]['QLIKE'] for y in years]), 'QLIKE')} for h in C.HS]),
         '3_8': pd.DataFrame(dm)}
    for metric in ('QLIKE', 'RMSE', 'MAE'):
        sfx = '' if metric == 'QLIKE' else '_' + metric.lower()
        one = lambda m, h: fmt(M[(m, h)][YEAR_ONE][metric], metric) if YEAR_ONE in M[(m, h)] else np.nan
        T['3_4' + sfx] = pd.DataFrame([{'Модель': m, **{f'H = {h}': one(m, h) for h in C.HS}} for m in ('XGBoost', 'LightGBM', 'HAR-J')])
        T['3_6' + sfx] = pd.DataFrame([{'Модель': m, **{f'H = {h}': one(m, h) for h in C.HS}} for m in ('Гибридная', 'LightGBM', 'XGBoost', 'HAR-J')])
        T['3_7' + sfx] = pd.DataFrame([{'Модель': m, **{f'H = {h}': fmt(np.mean([M[(m, h)][y][metric] for y in years]), metric) for h in C.HS}}
                                       for m in MODEL_ORDER])
    C.TABLES.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, df in T.items():
        f = C.TABLES / f'table_{n}_{tag}.csv'
        df.to_csv(f, index=False)
        rows.append(dict(файл=str(f.relative_to(C.ROOT)), строк=len(df), годы=span))
    return pd.DataFrame(rows)
