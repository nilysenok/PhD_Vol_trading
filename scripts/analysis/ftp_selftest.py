"""Проверка кода теста на синтетике (тестовые данные не читаются): python3 scripts/analysis/final_test.py --selftest"""
import tempfile
from pathlib import Path
import numpy as np, pandas as pd
import ftp_stats as S, ftp_tables as FT

COLS = ('har_ord', 'xgb_raw', 'lgb_raw', 'xgb_corr', 'lgb_corr')


def make(rng, nf=0.2, nh=0.6, start='2017-01-03', end='2025-12-26', nt=17):
    dates = pd.bdate_range(start, end)
    g = pd.DataFrame([(d, f'T{t}') for d in dates for t in range(nt)], columns=['date', 'ticker'])
    g['year'] = g.date.dt.year
    g['step'] = g.year - 2016
    g['rv_m'] = np.exp(rng.normal(-8, 0.5, len(g)))
    s = g.rv_m.values * np.exp(rng.normal(0, 0.3, len(g)))
    g['rv_actual'] = s * np.exp(rng.normal(0, 0.5, len(g)))
    g['rv_pred'] = s * np.exp(rng.normal(0, nf, len(g)))
    for c in ('har_raw', 'har_corr') + COLS:
        g[c] = s * np.exp(rng.normal(0, nh, len(g)))
    return g


def checks(rng):
    # 1. равные прогнозы: DM = 0; независимые одинаковые по качеству: |t| мал
    g = make(rng).assign(har_corr=lambda x: x.rv_pred)
    assert abs(S.dm_main(g, 'har_corr', 5)['t']) < 1e-9
    g2 = make(rng, 0.4, 0.4)
    assert abs(S.dm_main(g2, 'har_corr', 5)['t']) < 3.5
    # 2. MCS: равные по качеству остаются в наборе, заведомо худшая - вылетает; одинаковые прогнозы - обе в наборе
    L = np.column_stack([S.daily(g2.date.values, S.qloss(g2.rv_actual.values, g2[c].values)) for c in ('rv_pred', 'har_corr', 'har_raw')])
    for st in ('R', 'max'):
        p = S.mcs(L, ['a', 'b', 'c'], 5, B=2000, stat=st)
        assert min(p.values()) >= 0.10 and abs(max(p.values()) - 1) < 1e-12, p
        w = S.mcs(np.column_stack([L[:, :2], L[:, 0] + 0.5 + 0.3 * np.random.default_rng(1).normal(size=len(L))]), ['a', 'b', 'bad'], 5, B=2000, stat=st)
        assert w['bad'] < 0.01 and w['a'] > 0.10 and w['b'] > 0.10, w
        same = S.mcs(np.column_stack([L[:, 0], L[:, 0]]), ['a', 'b'], 5, B=500, stat=st)
        assert min(same.values()) == 1.0
    # 3. Минцер-Зарновиц: несмещённый прогноз - beta ~ 1, Wald не отвергает; сдвиг ×2 - beta ~ 0.5 в уровнях, Wald отвергает
    n, h = 4000, 5
    dts = np.repeat(np.arange(n // 10), 10)
    p = np.exp(rng.normal(0, 0.5, n))
    y = p + rng.normal(0, 0.3, n)
    ok, bad = S.mz(y, p, dts, h), S.mz(y, 2 * p, dts, h)
    assert abs(ok['beta'] - 1) < 0.1 and ok['p_wald'] > 0.01, ok
    assert abs(bad['beta'] - 0.5) < 0.1 and bad['p_wald'] < 1e-6, bad
    # 4. охват: B = A + шум в логах - A охватывает B, B не охватывает A
    ls = rng.normal(0, 0.5, 20000)
    dts = np.repeat(np.arange(2000), 10)
    eA = rng.normal(0, 0.3, 20000)
    w = rng.normal(0, 0.3, 20000)
    a_enc, b_enc = S.hln(eA, eA - w, dts, 5), S.hln(eA - w, eA, dts, 5)
    assert a_enc['p_one'] > 0.05 and abs(a_enc['lam']) < 0.1 and b_enc['p_one'] < 1e-3 and b_enc['lam'] > 0.3, (a_enc, b_enc)
    # 5. терцили без заглядывания: правка RV_m после даты t не меняет терциль на датах <= t; доли ~ 1/3
    f = make(rng, start='2016-01-04', end='2018-12-28')
    rows = f[f.date >= '2017-01-02'].reset_index(drop=True)
    t1 = S.vol_terciles(f[['date', 'ticker', 'rv_m']], rows)
    cut = pd.Timestamp('2017-07-03')
    f2 = f.copy()
    f2.loc[f2.date > cut, 'rv_m'] *= 50
    t2 = S.vol_terciles(f2[['date', 'ticker', 'rv_m']], rows)
    m = (rows.date <= cut).values
    assert (t1[m] == t2[m]).all() and abs(np.nanmean(t1 == 0) - 1 / 3) < 0.08


def main(run):
    rng = np.random.default_rng(0)
    checks(rng)
    tmp = Path(tempfile.mkdtemp())
    for h in (1, 5, 22):
        g = make(rng)
        g.drop(columns='rv_m').to_parquet(tmp / f'final_h{h}_test.parquet')
    feat = pd.concat([make(rng, start='2014-01-02', end='2016-12-30'), g])[['date', 'ticker', 'rv_m']]
    T = run(tmp, tmp / 'tables', 'selftest', feat=feat, B=1000)
    assert (T['3_8']['t-стат.'] > 0).all() and (T['3_9'].t > 0).all()
    rd = lambda n: pd.read_csv(tmp / 'tables' / f'table_3_{n}_selftest.csv', sep=';', decimal=',')
    mz, tk, sp, tc, mc, en = (rd(n) for n in ('mz', 'tickers', 'subperiods', 'terciles', 'mcs', 'encompassing'))
    for n in ('mz', 'tickers', 'subperiods', 'terciles', 'mcs', 'encompassing'):  # BOM и чтение без потерь
        f = tmp / 'tables' / f'table_3_{n}_selftest.csv'
        assert f.read_bytes()[:3] == b'\xef\xbb\xbf'
        d = rd(n)
        assert d.shape[1] > 3 and not d.columns.str.contains(';').any() and d.select_dtypes('number').shape[1] >= 2, n
    assert mz[['α', 'β', 'R²']].dtypes.eq(float).all() and sp.filter(like='QLIKE').dtypes.eq(float).all() and mc['p_MCS T_R'].dtype == float
    assert len(sp) == 12 and len(tc) == 9 and len(mc) == 12 and len(en) == 12 and len(mz) == 24 and len(tk) == 3 * 18
    assert (tk[tk['Тикер'].str.startswith('ДОЛЯ')]['финальная лучше'] == 1).all()  # финальная точнее HAR-J на синтетике у всех тикеров
    assert (sp['DM t vs HAR-J (б)'] > 0).all() and (tc['DM t vs HAR-J (б)'] > 0).all()
    assert (mc[mc['Модель'] == 'Финальная'].filter(like='в MCS').values == 1).all()  # лучшая модель всегда в наборе
    assert (mc[mc['Модель'] == 'HAR-J (б)']['p_MCS T_R'] < 0.1).all()  # заметно худшая - вне набора
    assert (en[(en.H0 == 'HAR-J (б) охватывает финальную') & (en.Шкала == 'ln RV')]['H0 отвергнута 5%'] == 1).all()  # слабая не охватывает сильную
    print('selftest OK:', tmp)
