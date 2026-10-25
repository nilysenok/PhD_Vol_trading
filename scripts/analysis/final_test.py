"""Слепой тест финального дизайна моделей (один раз, после «да» владельца на docs/otchety/FINAL_DESIGN.md).
python3 scripts/analysis/final_test.py --scope test            # падает, если нет docs/otchety/FINAL_DESIGN.md
python3 scripts/analysis/final_test.py --selftest              # проверка кода на синтетике (тест не читается)

Вход: <общий data/regen>/test_locked/final/final_h{h}_test.parquet (h = 1, 5, 22; собирает gibrid_v17.py --build-test):
date, ticker, year, step, rv_actual, rv_pred (финальный дизайн), har_raw (HAR-J как в диссертации), har_corr (HAR-J с Б0.1 той же оценки s²),
har_ord, xgb_raw/lgb_raw (компоненты ноги без поправки), xgb_corr/lgb_corr, design.
Фильтр строк: rv_actual > 1e-12 (как gibrid_io.merged).
Метрики: QLIKE = mean(y/p - ln(y/p) - 1), RMSE и MAE на уровне RV; по h и годам.
DM (потери QLIKE, d = L_эталон - L_финал, t > 0 - финальный лучше; эталоны: HAR-J как в диссертации и HAR-J с поправкой), три способа:
  (1) номинальный - простой t на пуле (дата, тикер); (2) NW (Бартлетт) с лагом h-1 на пуле, ряд упорядочен по (дата, тикер);
  (3) основной - дневной ряд средней по тикерам (17) разности потерь, NW с лагом h-1. H1 (финальный лучше HAR-J с поправкой) - по (3), одностороннее p.
Выход (docs/tables/, суффикс _test): table_3_2 (HAR-J по годам), 3_3, 3_4 (2019: XGBoost, LightGBM, HAR-J) + _rmse/_mae, 3_6 (2019: Гибридная, ...)
+ _rmse/_mae, 3_7 (среднее по годам) + _rmse/_mae, 3_8 (DM основной против HAR-J с поправкой), 3_9 (DM три способа × два эталона, H1),
table_3_final_po_godam_test.csv (финальный и эталоны по h и годам, QLIKE/RMSE/MAE). Табл. 3.1 (GJR-GARCH) и 3.5 (LSTM/GRU) здесь не строятся.
Колонки 3.2-3.8 - как models/tables.py (текст диссертации); бустинг в 3.4/3.6/3.7 - без поправки, как там; «Гибридная» - финальный дизайн."""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
MAIN = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading')
HS = (1, 5, 22)
YEAR_ONE = 2019
MODELS = {'Гибридная': 'rv_pred', 'HAR-J': 'har_raw', 'HAR-J с поправкой': 'har_corr', 'XGBoost': 'xgb_raw', 'LightGBM': 'lgb_raw',
          'XGBoost с поправкой': 'xgb_corr', 'LightGBM с поправкой': 'lgb_corr'}
REFS = {'HAR-J как в диссертации': 'har_raw', 'HAR-J с поправкой': 'har_corr'}


def qloss(y, p):
    r = y / p
    return r - np.log(r) - 1


def metrics(y, p):
    e = y - p
    return dict(QLIKE=float(qloss(y, p).mean()), RMSE=float(np.sqrt(np.mean(e ** 2))), MAE=float(np.mean(np.abs(e))))


def nw_var(d, lag):
    """Долгосрочная дисперсия среднего (Ньюи-Уэст, Бартлетт): (g0 + 2 sum (1 - k/(L+1)) g_k) / n."""
    d = np.asarray(d, float) - np.mean(d)
    n = len(d)
    g = d @ d / n
    for k in range(1, lag + 1):
        g += 2 * (1 - k / (lag + 1)) * (d[k:] @ d[:-k]) / n
    return g / n


def dm(df, ref, h):
    """Три способа DM; df отсортирован по (date, ticker). Возвращает список строк."""
    d = qloss(df.rv_actual.values, df[ref].values) - qloss(df.rv_actual.values, df.rv_pred.values)
    out = []
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    out.append(('номинальный (пул, без HAC)', n, d.mean() / se))
    out.append((f'NW лаг {h - 1} на пуле', n, d.mean() / np.sqrt(nw_var(d, h - 1))))
    daily = pd.Series(d, index=df.date.values).groupby(level=0).mean().sort_index().values
    out.append((f'основной: дневной ряд средней по тикерам, NW лаг {h - 1}', len(daily), daily.mean() / np.sqrt(nw_var(daily, h - 1))))
    return [dict(способ=m, n=k, t=float(t), p_two=float(2 * stats.norm.sf(abs(t))), p_one_H1=float(stats.norm.sf(t)), mean_d=float(d.mean()))
            for m, k, t in out]


def fmt(v, metric):
    return round(v, 4) if metric == 'QLIKE' else float(f'{v:.4g}')


def pfmt(p):
    return '<0,001' if p < 1e-3 else round(p, 3)


def run(src, tables, tag='test'):
    M, DM, rows = {}, [], []
    for h in HS:
        f = src / f'final_h{h}_test.parquet'
        df = pd.read_parquet(f)
        df = df[df.rv_actual > 1e-12].sort_values(['date', 'ticker']).reset_index(drop=True)
        miss = [c for c in MODELS.values() if c not in df]
        assert not miss, f'{f.name}: нет столбцов {miss}'
        for m, c in MODELS.items():
            for y, g in df.groupby('year'):
                M[(m, h, int(y))] = metrics(g.rv_actual.values, g[c].values)
                rows.append(dict(h=h, Модель=m, Год=int(y), **M[(m, h, int(y))]))
            rows.append(dict(h=h, Модель=m, Год='все годы (пул)', **metrics(df.rv_actual.values, df[c].values)))
        for rn, rc in REFS.items():
            for r in dm(df, rc, h):
                DM.append(dict(H=h, эталон=rn, **r))
    years = sorted({k[2] for k in M})
    span = f'{years[0]}-{years[-1]}'
    avg = lambda m, h, metric: np.mean([M[(m, h, y)][metric] for y in years if (m, h, y) in M])
    one = lambda m, h, metric: fmt(M[(m, h, YEAR_ONE)][metric], metric) if (m, h, YEAR_ONE) in M else np.nan
    T = {'3_2': pd.DataFrame([{'Год': y, **{f'QLIKE H = {h}': fmt(M[('HAR-J', h, y)]['QLIKE'], 'QLIKE') for h in HS}} for y in years]
                             + [{'Год': f'Среднее {span}', **{f'QLIKE H = {h}': fmt(avg('HAR-J', h, 'QLIKE'), 'QLIKE') for h in HS}}]),
         '3_3': pd.DataFrame([{'Горизонт': f'H = {h}', 'QLIKE': fmt(avg('HAR-J', h, 'QLIKE'), 'QLIKE')} for h in HS])}
    for metric in ('QLIKE', 'RMSE', 'MAE'):
        sfx = '' if metric == 'QLIKE' else '_' + metric.lower()
        T['3_4' + sfx] = pd.DataFrame([{'Модель': m, **{f'H = {h}': one(m, h, metric) for h in HS}} for m in ('XGBoost', 'LightGBM', 'HAR-J')])
        T['3_6' + sfx] = pd.DataFrame([{'Модель': m, **{f'H = {h}': one(m, h, metric) for h in HS}}
                                       for m in ('Гибридная', 'LightGBM', 'XGBoost', 'HAR-J', 'HAR-J с поправкой')])
        T['3_7' + sfx] = pd.DataFrame([{'Модель': m, **{f'H = {h}': fmt(avg(m, h, metric), metric) for h in HS}}
                                       for m in ('HAR-J', 'HAR-J с поправкой', 'XGBoost', 'LightGBM', 'Гибридная')])
    D = pd.DataFrame(DM)
    main = D[(D.эталон == 'HAR-J с поправкой') & D.способ.str.startswith('основной')]
    T['3_8'] = pd.DataFrame([{'H': r.H, 't-стат.': round(r.t, 2), 'p-value': pfmt(r.p_one_H1)} for r in main.itertuples()])
    T['3_9'] = D.assign(t=D.t.round(2), p_two=D.p_two.map(pfmt), p_one_H1=D.p_one_H1.map(pfmt),
                        H1_принята_5pct=[(e == 'HAR-J с поправкой' and s.startswith('основной') and p < 0.05) for e, s, p in zip(D.эталон, D.способ, D.p_one_H1)])
    T['3_final_po_godam'] = pd.DataFrame(rows)
    tables.mkdir(parents=True, exist_ok=True)
    for n, df in T.items():
        df.to_csv(tables / f'table_{n}_{tag}.csv', index=False)
        print('записано', tables / f'table_{n}_{tag}.csv', len(df), 'строк')
    return T


def selftest():
    """Синтетика: финальный лучше HAR-J -> DM t > 0 всеми способами; nw_var при лаге 0 = простая дисперсия среднего."""
    import tempfile
    rng = np.random.default_rng(0)
    tmp = Path(tempfile.mkdtemp())
    dates = pd.bdate_range('2017-01-03', '2018-12-28')
    for h in HS:
        g = pd.DataFrame([(d, t) for d in dates for t in range(17)], columns=['date', 'ticker'])
        g['year'] = g.date.dt.year
        g['step'] = g.year - 2016
        s = np.exp(rng.normal(-8, 0.5, len(g)))
        g['rv_actual'] = s * np.exp(rng.normal(0, 0.5, len(g)))
        g['rv_pred'] = s * np.exp(rng.normal(0, 0.2, len(g)))
        for c in ('har_raw', 'har_corr', 'har_ord', 'xgb_raw', 'lgb_raw', 'xgb_corr', 'lgb_corr'):
            g[c] = s * np.exp(rng.normal(0, 0.6, len(g)))
        g.to_parquet(tmp / f'final_h{h}_test.parquet')
    x = rng.normal(size=500)
    assert abs(nw_var(x, 0) - x.var() / len(x)) < 1e-12
    T = run(tmp, tmp / 'tables', 'selftest')
    assert (T['3_8']['t-стат.'] > 0).all() and (T['3_9'].t > 0).all()
    print('selftest OK:', tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scope', choices=['test'], help='test - один раз, после «да» на FINAL_DESIGN.md')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--src', default=str(MAIN / 'data/regen/test_locked/final'))
    ap.add_argument('--tables', default=str(ROOT / 'docs/tables'))
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.scope != 'test':
        sys.exit('нужен --scope test (или --selftest)')
    fd = ROOT / 'docs/otchety/FINAL_DESIGN.md'
    if not fd.exists():
        sys.exit(f'СТОП: нет {fd} - тест только после FINAL_DESIGN.md и «да» владельца')
    run(Path(a.src), Path(a.tables))


if __name__ == '__main__':
    main()
