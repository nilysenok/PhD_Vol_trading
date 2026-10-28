#!/usr/bin/env python3
"""Блок признаков Д10.1-Д10.3 (h=5 и h=22), строго point-in-time: признак даты t использует данные не позже 18:50 дня t.
build_d10(features_df) -> DataFrame (индекс как у features_df, колонки d10_*). Данные — в общем каталоге DATA (абсолютные пути).
Д10.1 состояние рынка: IMOEX (закрытие 18:50) просадка от 252-дн. максимума, доходности 22/66; средняя попарная корреляция 17 акций за 22 дня;
  кросс-секционная дисперсия доходностей (день, среднее за 22); RV RGBI, CNY-RUB, Brent за 22 дня и тренд Brent 66 (лаг 1 день: закрытие после 18:50).
Д10.2 режим: терцили RV_m (расширяющееся окно) и дней в режиме; дней с последнего скачка и число скачков 22/66 (скачок: J/RV >= 0,3, так как J > 0
  у 83% дней); RVI(18:50) - 100*RV_IMOEX_22; фильтрованная вероятность высокого режима MarkovRegression на ln RV (параметры - до начала года, фильтр вперёд).
Д10.3 цикл ставки: направление последнего изменения, число изменений за 182 дня, дней с последнего заседания, дней до планового (known_from <= t), маркеры."""
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'
import warnings, multiprocessing as mp
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')
DATA = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data')
OUT = DATA / 'regen' / 'd10'
CAND = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/moex_discovery/data/final_v6/candles_10m')
RVI10 = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/moex_discovery/data/external/moex_iss/indices_10m/RVI.parquet')
CUT = 18 * 60 + 50
JUMP_SHARE = 0.3
D10_EXCLUDE = ['d10_days_after_20220224', 'd10_days_after_20200622', 'd10_ms_p_high']  # нулевой суммарный gain шагов 1-6 (h=5 и h=22), см. отчёт


def _cut_last(path, col='close'):
    """Закрытие дня по последней 10-минутной свече с началом < 18:50."""
    c = pd.read_parquet(path)
    ts = pd.to_datetime(c['timestamp'] if 'timestamp' in c else c['begin'])
    ok = (ts.dt.hour * 60 + ts.dt.minute) < CUT
    return c.loc[ok, col].groupby(ts[ok].dt.normalize()).last()


def prices_1850(cache=True):
    """Цены 18:50: DataFrame дата x (17 акций + IMOEX); кэш data/regen/d10/close1850.parquet."""
    p = OUT / 'close1850.parquet'
    if cache and p.exists():
        return pd.read_parquet(p)
    cols = {f.stem: _cut_last(f) for f in sorted((CAND / 'stocks').glob('*.parquet'))}
    cols['IMOEX'] = _cut_last(CAND / 'external/IMOEX.parquet')
    P = pd.DataFrame(cols).sort_index()
    if cache:
        OUT.mkdir(parents=True, exist_ok=True); P.to_parquet(p)
    return P


def rvi_1850(path=RVI10):
    return _cut_last(path)


def _by_date(A, cols):
    return A.groupby('date')[list(cols)].first().sort_index().astype(float)


def f_state(A, P, ud):
    """Д10.1: ряды уровня даты (индекс ud)."""
    P = P.dropna(subset=['IMOEX']).sort_index()
    imx = P['IMOEX']
    R = np.log(P.drop(columns='IMOEX')).diff()
    R = R.where(R.abs() < 0.4)  # консолидации/сплиты -> пропуск
    o = pd.DataFrame(index=P.index)
    o['imoex_dd252'] = imx / imx.rolling(252, min_periods=120).max() - 1
    o['imoex_ret22'] = np.log(imx / imx.shift(22))
    o['imoex_ret66'] = np.log(imx / imx.shift(66))
    o['xs_disp1'] = R.std(axis=1)
    o['xs_disp22'] = o['xs_disp1'].rolling(22, min_periods=11).mean()
    v, pc = R.values, np.full(len(R), np.nan)
    for i in range(21, len(R)):
        w = v[i - 21:i + 1]
        ok = (~np.isnan(w)).sum(0) >= 15
        if ok.sum() >= 5:
            C = pd.DataFrame(w[:, ok]).corr().values
            pc[i] = np.nanmean(C[np.triu_indices_from(C, 1)])
    o['pair_corr22'] = pc
    o = o.reindex(ud, method='ffill')
    M = _by_date(A, ['idx_RGBI_rv_annualized_lag1', 'idx_CNYRUB_TOM_rv_annualized_lag1', 'yahoo_brent_lag1']).reindex(ud).ffill()
    o['rv22_rgbi'] = M['idx_RGBI_rv_annualized_lag1'].rolling(22, min_periods=11).mean()  # lag1: ряд на t = RV дня t-1
    o['rv22_cnyrub'] = M['idx_CNYRUB_TOM_rv_annualized_lag1'].rolling(22, min_periods=11).mean()
    b = M['yahoo_brent_lag1']
    o['brent_rv22'] = np.log(b).diff().rolling(22, min_periods=11).std()
    o['brent_trend66'] = np.log(b / b.shift(66))
    return o


def f_rate(A, ud, cal):
    """Д10.3 (индекс ud). cal - DataFrame cbr.csv."""
    kr = _by_date(A, ['key_rate_cbr'])['key_rate_cbr'].ffill().reindex(ud).ffill()
    d = kr.diff()
    o = pd.DataFrame(index=ud)
    o['kr_last_dir'] = np.sign(d.where(d != 0)).ffill().fillna(0)
    o['kr_n_chg_6m'] = (d.fillna(0) != 0).astype(float).rolling('182D').sum()
    held = np.sort(pd.to_datetime(cal['date']).values)  # заседание с датой <= t состоялось
    ix = ud.values
    pos = np.searchsorted(held, ix, side='right') - 1
    o['days_since_cbr'] = np.where(pos >= 0, (ix - held[np.clip(pos, 0, None)]) / np.timedelta64(1, 'D'), np.nan)
    pl = cal[cal['vneplan'].fillna(0).astype(int) == 0]
    dt, kf = pd.to_datetime(pl['date']).values, pd.to_datetime(pl['known_from']).values
    o['days_to_plan'] = [((dt[(kf <= t) & (dt > t)].min() - t) / np.timedelta64(1, 'D')) if ((kf <= t) & (dt > t)).any() else np.nan for t in ix]
    for nm, dd in (('days_after_20220224', '2022-02-24'), ('days_after_20200622', '2020-06-22')):
        o[nm] = np.clip((ix - np.datetime64(dd)) / np.timedelta64(1, 'D'), 0, None)
    return o


def _runlen(s):
    return s.groupby(s.ne(s.shift()).cumsum()).cumcount() + 1


def f_regime(A, s1850):
    """Д10.2 без марковской модели (индекс как A; A отсортирован по дате)."""
    g = A.groupby('ticker')
    pct = g['rv_m'].transform(lambda s: s.expanding(min_periods=60).rank(pct=True))
    reg = pd.Series(np.where(pct.isna(), np.nan, np.floor(np.clip(pct, 0, 0.9999) * 3)), index=A.index)
    o = pd.DataFrame(index=A.index)
    o['rvm_regime'] = reg
    o['rvm_regime_days'] = reg.groupby(A['ticker']).transform(_runlen).where(reg.notna())
    jf = ((A['jump'] / A['rv']) >= JUMP_SHARE).astype(float)
    cnt = jf.groupby(A['ticker']).cumcount().astype(float)
    o['days_since_jump'] = cnt - cnt.where(jf > 0).groupby(A['ticker']).ffill()  # торговых дней тикера с последнего скачка (NaN до первого)
    o['n_jumps22'] = jf.groupby(A['ticker']).transform(lambda s: s.rolling(22, min_periods=11).sum())
    o['n_jumps66'] = jf.groupby(A['ticker']).transform(lambda s: s.rolling(66, min_periods=33).sum())
    ud = pd.DatetimeIndex(sorted(A['date'].unique()))
    rvi = s1850.reindex(ud, method='ffill')
    imx = _by_date(A, ['rv_IMOEX'])['rv_IMOEX'].reindex(ud).ffill().shift(1).rolling(22, min_periods=11).mean()  # лаг 1 день
    o['rvi_minus_rvimoex22'] = A['date'].map(rvi - 100 * imx).values
    return o


def _ms_one(args):
    """Фильтрованная P(высокий режим) одного тикера: для года Y параметры по данным < Y (>= 250 дней), фильтр по данным <= Y (причинный)."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression as MR
    tk, dts, y, yrs = args
    res = []
    for Y in sorted(set(yrs)):
        fit = yrs < Y
        if fit.sum() < 250:
            continue
        lo, hi = np.percentile(y[fit], [0.5, 99.5])  # винзоризация и стандартизация - по данным < Y
        z = (np.clip(y, lo, hi) - y[fit].mean()) / y[fit].std()
        np.random.seed(Y)
        try:
            m = MR(z[fit], k_regimes=2, trend='c', switching_variance=True)
            r = m.fit(disp=False, search_reps=5)
            if not r.mle_retvals.get('converged', True) or r.expected_durations.min() < 2:  # не сошлась / вырожденный режим
                continue
            hi_r = int(np.argmax(r.params[[i for i, n in enumerate(m.param_names) if n.startswith('const')]]))
            up = yrs <= Y
            pr = np.asarray(MR(z[up], k_regimes=2, trend='c', switching_variance=True).filter(r.params).filtered_marginal_probabilities)[:, hi_r]
        except Exception:
            continue
        sel = yrs[up] == Y
        res.append(pd.DataFrame({'date': dts[up][sel], 'ticker': tk, 'ms_p_high': pr[sel]}))
    return pd.concat(res) if res else pd.DataFrame(columns=['date', 'ticker', 'ms_p_high'])


def markov_p(A, workers=2):
    jobs = []
    for tk, g in A[['date', 'ticker', 'rv']].groupby('ticker'):
        jobs.append((tk, g['date'].values, np.log(g['rv'].clip(lower=1e-10).values), pd.DatetimeIndex(g['date']).year.values))
    if workers > 1:
        with mp.get_context('fork').Pool(workers) as p:
            out = p.map(_ms_one, jobs, chunksize=1)
    else:
        out = [_ms_one(j) for j in jobs]
    return pd.concat(out, ignore_index=True)


def build_d10(features_df, prices=None, rvi=None, cal=None, ms=None, workers=2):
    """features_df: матрица features_pit_v2 (date, ticker, rv, rv_m, jump, ...). ms - готовые фильтрованные вероятности (иначе считаются)."""
    A = features_df.sort_values(['date', 'ticker'], kind='stable') if not features_df['date'].is_monotonic_increasing else features_df
    ud = pd.DatetimeIndex(sorted(A['date'].unique()))
    P = prices_1850() if prices is None else prices
    cal = pd.read_csv(DATA / 'calendar/cbr.csv') if cal is None else cal
    rvi = rvi_1850() if rvi is None else rvi
    out = pd.DataFrame(index=A.index)
    for G in (f_state(A, P, ud), f_rate(A, ud, cal)):
        for c in G.columns:
            out[f'd10_{c}'] = A['date'].map(G[c]).values
    R = f_regime(A, rvi)
    for c in R.columns:
        out[f'd10_{c}'] = R[c].values
    ms = markov_p(A, workers) if ms is None else ms
    out['d10_ms_p_high'] = A[['date', 'ticker']].merge(ms, on=['date', 'ticker'], how='left')['ms_p_high'].values
    return out.reindex(features_df.index)


if __name__ == '__main__':
    F = pd.read_parquet(DATA / 'regen/features_pit_v2.parquet')
    D = build_d10(F)
    R = pd.concat([F, D], axis=1)
    R.to_parquet(DATA / 'regen/features_pit_v2_d10.parquet')
    print(D.shape, D.notna().mean().round(3).to_dict())
