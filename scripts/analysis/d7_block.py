"""Блок признаков Д7 (h=5, h=22), point-in-time: значение на дате t использует только строки матрицы с date <= t.
Вход - data/regen/features_pit_v2.parquet (внешние ряды там уже с лагом доступности: Yahoo лаг 1, макро as-of, индексы MOEX срез 18:50).
  Д7.1 агрегаты внешних рядов под горизонт (окно w=h): среднее, изменение, СКО дневных приращений, отношение w/m (5/22);
  Д7.2 глобальный фактор волатильности: PC1 по 5-дневной реализованной волатильности глобальных активов (+ log VIX), PCA на
       расширяющемся окне, переоценка 1 января каждого года по данным строго до этой даты; признаки - уровень и 22-дневное изменение;
  Д7.3 календарь (d73_features.py): дни до события и число событий в [t+1, t+h_кал] среди известных на t (known_from <= t);
  Д7.4 RVI: уровень (срез 18:50), годовой перцентиль (окно 252), RVI / RV_IMOEX_22.
Строки-заполнители (медиана вместо отсутствующих внешних данных: до 26.08.2014 и выходные торговые дни) исключаются из окон
и получают значения последнего реального дня. Запуск: python3 d7_block.py  -> data/regen/features_pit_v2_d7_h{5,22}.parquet"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d73_features as d73

DATA = os.environ.get('PHD_DATA', '/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data')
YAHOO = {'brent': 'p', 'sp500': 'p', 'gold': 'p', 'dxy': 'p', 'eem': 'p', 'wti': 'p', 'vix': 'v',
         'tbill_3m': 'r', 'treasury_10y': 'r', 'treasury_5y': 'r'}
IDX = ['MOEXMM', 'IMOEX', 'MOEXTL', 'MOEXEU', 'CNYRUB_TOM', 'RVI', 'MOEXCH', 'MOEXFN', 'MOEXCN', 'RGBI', 'MOEXOG']
MACRO = {'cpi_ru': 'p', 'cpi_us': 'p', 'm2_ru': 'p', 'si_fut': 'p', 'unemp_us': 'r', 'pmi_ru': 'r', 'key_rate_cbr': 'r',
         'fed_rate': 'r'}
KINDS = dict(YAHOO, **MACRO)  # p: цена (лог), v: положительный уровень-волатильность (лог), r: ставка/индекс (без лога)
for x in IDX:
    KINDS['close_' + x] = 'v' if x == 'RVI' else 'p'
    KINDS['rv_' + x] = 'v'
GLOBAL = ['sp500', 'eem', 'dxy', 'gold', 'brent', 'wti']  # глобальные активы для фактора волатильности
CAL_TYPES = ['cbr', 'fomc', 'cpi_us', 'expirations', 'imoex_rebal']  # ИПЦ РФ и дивиденды отсутствуют
CAL_DAYS = {5: 7, 22: 31}  # h торговых дней -> календарных дней окна [t+1, t+h_кал]
REF = 'close_IMOEX'  # по нему узнаём строки-заполнители (константа первой строки матрицы)


def daily_levels():
    """Дневные уровни внешних рядов (исключаются флагом drop_daily): уровень, lag1, ret1/diff1."""
    c = []
    for s in YAHOO: c += [s, 'yahoo_%s_lag1' % s, 'yahoo_%s_ret1' % s]
    for x in IDX: c += ['close_' + x, 'rv_' + x, 'idx_%s_rv_annualized_lag1' % x, 'idx_%s_rv_annualized_ret1' % x]
    for m in MACRO: c += [m, m + '_lag1', m + '_diff1']
    return c


def _rank_last(a):
    return float((a <= a[-1]).mean())


def _agg(x, kind, w, name):
    """Д7.1 для одного ряда x (индекс - реальные торговые дни)."""
    z = x if kind == 'r' else np.log(x.clip(lower=1e-10))
    mp = max(2, int(0.6 * w))
    mean_w = z.rolling(w, min_periods=mp).mean()
    r = {'mean%d' % w: (z - mean_w) if kind == 'p' else mean_w,  # для цен - отклонение от среднего (стационарно)
         'chg%d' % w: z - z.shift(w), 'std%d' % w: z.diff().rolling(w, min_periods=mp).std()}
    a5, a22 = x.rolling(5, min_periods=3).mean(), x.rolling(22, min_periods=13).mean()
    r['wm'] = (a5 - a22) if kind == 'r' else a5 / a22 - (1 if kind == 'p' else 0)
    return pd.DataFrame({'d7_%s_%s' % (name, k): v for k, v in r.items()})


def _gvol(v):
    """Д7.2: PC1, веса и стандартизация оцениваются 1 января года Y по строкам строго до этой даты."""
    r = {a: np.log(v[a].clip(lower=1e-10)).diff() for a in GLOBAL}
    X = pd.DataFrame({'rv_' + a: np.log(np.sqrt((r[a] ** 2).rolling(5, min_periods=4).mean()) + 1e-4) for a in GLOBAL})
    X['lvix'] = np.log(v['vix'].clip(lower=1e-10))
    pc = pd.Series(np.nan, index=v.index)
    for y in sorted(set(v.index.year)):
        fit = X[X.index < pd.Timestamp(y, 1, 1)].dropna()
        if len(fit) < 252:
            continue
        mu, sd = fit.mean(), fit.std().replace(0, np.nan)
        w = np.linalg.svd(((fit - mu) / sd).values, full_matrices=False)[2][0]
        w = w if w.sum() >= 0 else -w
        m = (v.index.year == y) & X.notna().all(axis=1).values
        pc[m] = (((X[m] - mu) / sd).values * w).sum(axis=1)
    return pd.DataFrame({'d7_gvol_pc1': pc, 'd7_gvol_pc1_chg22': pc - pc.shift(22)})


def _rvi(v):
    """Д7.4: RVI на срезе 18:50, годовой перцентиль, RVI / RV_IMOEX_22 (RV_IMOEX - годовая вол. в долях, NaN в дни с неполными барами)."""
    rvi = v['close_RVI']
    rv22 = v['rv_IMOEX'].rolling(22, min_periods=13).mean()
    return pd.DataFrame({'d7_rvi': rvi, 'd7_rvi_pctl252': rvi.rolling(252, min_periods=126).apply(_rank_last, raw=True),
                         'd7_rvi_rvimoex22': rvi / (100 * rv22)})


def date_block(x, h, cal_dir=None):
    """x - DataFrame внешних колонок, индекс - даты (каждая один раз). Возвращает d7_-признаки по датам."""
    fill = (x[REF] == x[REF].iloc[0]).values
    v = x[~fill].copy()
    v.loc[v['imoex_low_bars'] == 1, 'rv_IMOEX'] = np.nan
    v['rv_IMOEX'] = v['rv_IMOEX'].ffill(limit=2)
    parts = [_agg(v[c], k, h, c) for c, k in KINDS.items() if c in v]
    parts += [_gvol(v), _rvi(v)]
    out = pd.concat(parts, axis=1).reindex(x.index, method='ffill')
    ev = d73.load_events(cal_dir or os.path.join(DATA, 'calendar'), CAL_TYPES)
    cal = d73.features_frame(x.index, CAL_DAYS[h], ev).add_prefix('d7_cal_')
    return pd.concat([out, cal], axis=1)


def build_d7(features_df, h, cal_dir=None):
    """Новые колонки d7_* (индекс = индекс features_df); h in {5, 22}. Внутри считается по уникальным датам."""
    ext = sorted(set(KINDS) | {'imoex_low_bars'})
    x = features_df[['date'] + [c for c in ext if c in features_df]].drop_duplicates('date').sort_values('date').set_index('date')
    d = date_block(x, h, cal_dir)
    out = d.reindex(features_df['date'].values)
    out.index = features_df.index
    return out.astype('float64')


def assemble(features_df, h, drop_daily=True, cal_dir=None):
    """Матрица для модели: исходная (без дневных уровней внешних рядов, если drop_daily) + d7_-колонки."""
    base = features_df.drop(columns=[c for c in daily_levels() if c in features_df]) if drop_daily else features_df
    return pd.concat([base, build_d7(features_df, h, cal_dir)], axis=1)


if __name__ == '__main__':
    df = pd.read_parquet(os.path.join(DATA, 'regen', 'features_pit_v2.parquet'))
    for h in (5, 22):
        m = assemble(df, h)
        m.to_parquet(os.path.join(DATA, 'regen', 'features_pit_v2_d7_h%d.parquet' % h), index=False)
        print('h=%d: %s, d7-колонок %d' % (h, m.shape, sum(c.startswith('d7_') for c in m)), flush=True)
