"""Загрузка прогнозов бустинга (single / top5 / seeds3) и слияние с HAR-J."""
from pathlib import Path
import pandas as pd
import gibrid_har as GH

MAIN_REGEN = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen')


def find(regen, name):
    for d in (regen, GH.ROOT / 'data/regen', MAIN_REGEN):
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(name)


def _avg(df):  # ансамбль: среднее прогнозов по (date, ticker, step)
    return df.groupby(['date', 'ticker', 'step'], as_index=False).agg(
        year=('year', 'first'), rv_actual=('rv_actual', 'first'), rv_pred=('rv_pred', 'mean'))


def load_boost(boost, h, regen, variant):
    """single — одна точка; top5 — среднее топ-5 (B_*_h{h}_top5_*); seeds3 — seed 42 + B_*_h{h}_seeds4344_* (формат tune_B.py)."""
    res = []
    closed = boost not in ('P1', 'B')  # Bv2, Bv2b, L2_h* (уровень 2): тест закрыт
    for n in {'P1': ('xgb', 'P1'), 'B': ('B_xgb', 'B_lgbm')}.get(boost, (f'{boost}_xgb', f'{boost}_lgbm')):
        d = {}
        for k in ('val', 'test'):
            if k == 'test' and closed:  # тест B v2 закрыт (data/regen/test_locked, не открывать): только валидация
                d[k] = pd.DataFrame(columns=['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']).astype(
                    {'date': 'datetime64[ns]', 'ticker': object, 'year': int, 'step': int, 'rv_actual': float, 'rv_pred': float})
                continue
            if variant == 'single':
                d[k] = pd.read_parquet(find(regen, f'{n}_h{h}_{k}.parquet'))
            elif variant == 'top5':
                d[k] = _avg(pd.read_parquet(find(regen, f'{n}_h{h}_top5_{k}.parquet')))
            else:
                d[k] = _avg(pd.concat([pd.read_parquet(find(regen, f'{n}_h{h}_{k}.parquet')),
                                       pd.read_parquet(find(regen, f'{n}_h{h}_seeds4344_{k}.parquet'))]))
        res.append(d)
    return res


def merged(har, xgb, lgb):
    m = har.rename(columns={'rv_pred': 'har'})
    for n, d in (('xgb', xgb), ('lgb', lgb)):
        m = m.merge(d[['date', 'ticker', 'rv_pred']].rename(columns={'rv_pred': n}), on=['date', 'ticker'])
    return m[m['rv_actual'] > 1e-12].sort_values(['step', 'date', 'ticker']).reset_index(drop=True)  # фильтр: v3.py:65
