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


def _read(regen, name, test_dir):
    if test_dir is not None:  # тестовые прогнозы закрытых вариантов: data/regen/test_locked/<boost>/ (только сборка, без метрик)
        for d in (test_dir, MAIN_REGEN / 'test_locked'):
            if (d / name).exists():
                return pd.read_parquet(d / name)
        raise FileNotFoundError(name)
    return pd.read_parquet(find(regen, name))


def load_boost(boost, h, regen, variant, test=False):
    """single — одна точка; top5 — среднее топ-5 (B_*_h{h}_top5_*; на шагах без топ-5 (7-9: точка шага 6 без перебора) — single,
    как sigma_pred_h1_v2); seeds3 — seed 42 + B_*_h{h}_seeds4344_* (формат tune_B.py);
    wrap — обёртка уровня 2 (wrap_v9.py, h=5): XGB-слот = V9-XGB (single), LGB-слот = обёртка = среднее V9-LGBM и V9-XGB по RV
    до поправки (совпадает с wrap_h5_val.parquet / test_locked/wrap до 5e-15; на шагах 7-9 - то же среднее точек шага 6).
    test=True — тест закрытых вариантов читается из test_locked (только для сборки тестовых прогнозов выбранного дизайна)."""
    closed = boost not in ('P1', 'B')  # Bv2, Bv2b, L2_h* (уровень 2): тест закрыт
    names = {'P1': ('xgb', 'P1'), 'B': ('B_xgb', 'B_lgbm')}.get(boost, (f'{boost}_xgb', f'{boost}_lgbm'))
    if boost.startswith(('GRU_sv', 'GARCH_sv')):  # одиночная модель (GRU_sv, GRU_svoos: {boost}_h{h}_{val,test}.parquet) в обоих слотах; V0_XGB = V0_LGB = модель
        names = (boost, boost)
    tdir = MAIN_REGEN / 'test_locked' / ('GARCH_sv' if boost.startswith('GARCH_sv') else boost)
    empty = lambda: pd.DataFrame(columns=['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']).astype(
        {'date': 'datetime64[ns]', 'ticker': object, 'year': int, 'step': int, 'rv_actual': float, 'rv_pred': float})
    res = []
    for n in names:
        d = {}
        for k in ('val', 'test'):
            if k == 'test' and closed and not test:  # тест закрыт (data/regen/test_locked, не открывать): только валидация
                d[k] = empty(); continue
            td = tdir if (k == 'test' and closed) else None
            rd = lambda nm: _read(regen, nm, td)
            single = rd(f'{n}_h{h}_{k}.parquet')
            if variant in ('single', 'wrap'):
                d[k] = single
            elif variant == 'wrap2':  # обёртка второго круга (final2_wrap.py): одинаковая для всех кандидатов
                d[k] = rd(f'{n}_h{h}_wrap2_{k}.parquet')
            elif variant == 'top5':
                t5 = _avg(rd(f'{n}_h{h}_top5_{k}.parquet'))
                miss = single[~single.step.isin(t5.step.unique())]
                d[k] = pd.concat([t5, miss[t5.columns]], ignore_index=True)
            else:
                d[k] = _avg(pd.concat([single, rd(f'{n}_h{h}_seeds4344_{k}.parquet')]))
        res.append(d)
    if variant == 'wrap':
        bx, bl = res
        w = {}
        for k in ('val', 'test'):
            m = bx[k].merge(bl[k][['date', 'ticker', 'step', 'rv_pred']], on=['date', 'ticker', 'step'], suffixes=('_x', '_l'))
            w[k] = m.assign(rv_pred=(m.rv_pred_x + m.rv_pred_l) / 2)[bx[k].columns.intersection(['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred'])]
        res = [bx, w]
    return res


def merged(har, xgb, lgb):
    m = har.rename(columns={'rv_pred': 'har'})
    for n, d in (('xgb', xgb), ('lgb', lgb)):
        m = m.merge(d[['date', 'ticker', 'rv_pred']].rename(columns={'rv_pred': n}), on=['date', 'ticker'])
    return m[m['rv_actual'] > 1e-12].sort_values(['step', 'date', 'ticker']).reset_index(drop=True)  # фильтр: v3.py:65
