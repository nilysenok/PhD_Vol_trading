"""Уровень 2 бустинга (Д1-Д8, М2): общая библиотека — признаки рычагов, данные шага, обучение одной точки, оценка Б0.1 вне выборки.
Все данные и артефакты — в общем каталоге DATA (абсолютный путь). Тест (шаги 1-9, тестовые годы) здесь не оценивается;
подтверждающий набор (val шагов 7-9) не считается.

Конфигурация рычагов cfg (dict; пусто = точка B без рычагов):
  obj   'mse' (ln RV, l2) | 'gamma' (уровень RV, лог-связь; XGB reg:gamma, LGBM gamma) — Д1
  hl    полураспад весов по давности, лет (None = без весов) — Д2
  cs    кросс-секционные признаки дня t по rv, rv_w, rv_m — Д3
  cat   категориальные тикер и сектор — Д4
  drop  {model: [признаки]} — отбор по суммарному gain шагов 1-6 (после Д3-Д4)
  mono  monotone_constraints +1 на rv, rv_d, rv_w, rv_m, bv — Д5
  cal   календарь событий [t+1, t+h] (Д7.3), known_from <= t, без ИПЦ РФ и дивидендов
  v9    бустинг на остатках HAR-J: смещение ln HAR-J (Д7.5)
  agg   агрегаты внешних рядов под горизонт, дневные уровни исключены (Д7.1)
  pc1   PC1 по волатильности глобальных рядов, расширяющееся окно (Д7.2)
  rvi   RVI со срезом 18:50 (Д7.4)
  harf  ln прогноза HAR-J как признак (Д8.12)
  avg2  среднее прогнозов моделей шагов t и t-1 (Д8.4, постобработка)
  h1f   ln прогноза h=1 (XGB, точка B h=1, OOF по годам на обучающем окне) как признак (Д8.6)
  wins  винзоризация ln RV цели по тикеру 1/99 на обучающем окне (Д6)
  hadd  добавки по горизонтам (Д6; h=22: RV 44/66, RV_w/RV_m, RV IMOEX 22, макро 22, СКО RV 22; h=5: RVI и его изменение за 5 дней, диапазон 5 дней)
  multi одна модель на три h с признаком h (Д6)
"""
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')
DATA = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data')
REGEN = DATA / 'regen'
L2DIR = REGEN / 'L2'
FEAT = Path(os.environ.get('L2_FEAT', str(REGEN / 'features_pit_v2.parquet')))  # Д12: L2_FEAT=<...>/features_pit_v3.parquet (as-of)
FEAT_CFG = FEAT.with_suffix('.config.json') if FEAT.with_suffix('.config.json').exists() else REGEN / 'features_pit_v2.config.json'
CAL = DATA / 'calendar'
RVI10 = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/moex_discovery/data/external/moex_iss/indices_10m/RVI.parquet')
TEST_YEARS = list(range(2017, 2026))  # шаг k -> тестовый год 2016+k; val = год перед тестовым
PAT, NMAX = 30, 1000
MAX_BIN = int(os.environ.get('L2_MAXBIN', 255))  # (63 — только для сверки с Bv2: LGBM 63, XGB 64) решение 25.10 (после проверки max_bin 63 владельцем): B v2b и уровень 2 — 255
SECTOR = {'LKOH': 'og', 'NVTK': 'og', 'TATN': 'og', 'SBER': 'fn', 'VTBR': 'fn', 'MOEX': 'fn', 'HYDR': 'eu', 'IRAO': 'eu', 'OGKB': 'eu',
          'ALRS': 'mm', 'MTLR': 'mm', 'MTSS': 'tl', 'RTKM': 'tl', 'MGNT': 'cn', 'PHOR': 'ch', 'AFLT': 'tn', 'LSRG': 're'}  # по отраслевым индексам MOEX
MONO = ('rv', 'rv_d', 'rv_w', 'rv_m', 'bv')
CAL_TYPES = ('cbr', 'fomc', 'cpi_us', 'expirations', 'imoex_rebal')  # без ИПЦ РФ и дивидендов (нет данных; решение по PR #13)
MOEX_IDX = ('MOEXMM', 'IMOEX', 'MOEXTL', 'MOEXEU', 'CNYRUB_TOM', 'RVI', 'MOEXCH', 'MOEXFN', 'MOEXCN', 'RGBI', 'MOEXOG')
YAHOO = ('brent', 'vix', 'sp500', 'gold', 'dxy', 'eem', 'wti', 'tbill_3m', 'treasury_10y', 'treasury_5y')
MACRO = ('cpi_ru', 'cpi_us', 'unemp_us', 'm2_ru', 'pmi_ru', 'key_rate_cbr', 'fed_rate', 'si_fut')
GLOBAL_VOL = ('yahoo_sp500_vol22', 'yahoo_eem_vol22', 'yahoo_gold_vol22', 'yahoo_wti_vol22', 'yahoo_brent_vol22', 'yahoo_dxy_vol22', 'vix')


def tcol(h):
    return f'rv_target_h{h}' if h == 1 else f'rv_target_mean_h{h}'  # h=5/22: ln среднего RV за [t+1, t+h]


def clean(X):
    return np.nan_to_num(X, nan=0, posinf=0, neginf=0)


def qlike(y, f):
    f = np.clip(f, 1e-10, None)
    return float(np.mean(y / f - np.log(y / f) - 1))


# ---------------------------------------------------------------- признаки рычагов (все — из данных <= t)
D10_EXCLUDE = ('d10_days_after_20220224', 'd10_days_after_20200622', 'd10_ms_p_high')  # по gain, d10_block.py (PR #28)


def load_base(matrix=None, blocks=(), h=None):
    """blocks: 'd7' — матрица features_pit_v2_d7_h{h}.parquet (те же строки; дневные уровни внешних рядов убраны, добавлены d7_*; PR #27);
    'd10' — столбцы d10_* из features_pit_v2_d10.parquet (PR #28) без D10_EXCLUDE. Признаки = признаки features_pit_v2, оставшиеся в матрице,
    + d7_* + d10_*. matrix — явный путь к альтернативной матрице (как d7)."""
    cfg = json.load(open(FEAT_CFG))
    if 'd7' in blocks:
        matrix = REGEN / f'{FEAT.stem}_d7_h{h}.parquet'  # v2: features_pit_v2_d7_h{h}; v3: features_pit_v3_d7_h{h}
    A = pd.read_parquet(matrix or FEAT).sort_values('date').reset_index(drop=True)  # порядок строк как в tune_Bv2.py
    if 'd10' in blocks:
        D10 = pd.read_parquet(REGEN / 'features_pit_v2_d10.parquet')
        c10 = [c for c in D10.columns if c.startswith('d10_') and c not in D10_EXCLUDE]
        A = A.merge(D10[['date', 'ticker'] + c10], on=['date', 'ticker'], how='left', validate='one_to_one')
    A['year'] = pd.to_datetime(A['date']).dt.year
    fc = [c for c in cfg['feature_cols'] if c in A.columns] + [c for c in A.columns if c.startswith(('d7_', 'd10_'))]
    return A, fc


def by_date(A, cols):
    """Ряды уровня даты (внешние ряды одинаковы у всех тикеров даты): первое непустое значение."""
    return A.groupby('date')[list(cols)].first().sort_index()


def f_cs(A):  # Д3: дня t по 17 тикерам (только строки той же даты)
    out = {}
    for c in ('rv', 'rv_w', 'rv_m'):
        g = A.groupby('date')[c]
        mu = g.transform('mean')
        out[f'cs_mean_{c}'] = mu
        out[f'cs_rank_{c}'] = g.rank(pct=True)
        out[f'cs_rel_{c}'] = A[c] / mu
        out[f'cs_var_{c}'] = g.transform('var')
    return pd.DataFrame(out, index=A.index)


def f_cat(A):  # Д4
    tick = {t: i for i, t in enumerate(sorted(SECTOR))}
    sec = {s: i for i, s in enumerate(sorted(set(SECTOR.values())))}
    return pd.DataFrame({'cat_ticker': A['ticker'].map(tick).astype(float), 'cat_sector': A['ticker'].map(SECTOR).map(sec).astype(float)}, index=A.index)


def load_calendar(types=CAL_TYPES, cal_dir=CAL):
    ev = {}
    for ty in types:
        d = pd.read_csv(cal_dir / f'{ty}.csv')
        if 'vneplan' in d:
            d = d[d['vneplan'].fillna(0).astype(int) == 0]
        ev[ty] = (pd.to_datetime(d['date']).values, pd.to_datetime(d['known_from']).values)
    return ev


def f_cal(dates, h, ev):
    """Д7.3: число известных на t событий в [t+1, t+W] (W = round(h*7/5) календарных дней) и дней до ближайшего известного."""
    W = np.timedelta64(int(round(h * 7 / 5)), 'D')
    ud = np.array(sorted(pd.unique(pd.to_datetime(dates))), dtype='datetime64[ns]')
    res = {}
    for ty, (d, kf) in ev.items():
        n, dte = np.zeros(len(ud)), np.full(len(ud), np.nan)
        for i, t in enumerate(ud):
            k = d[kf <= t]
            fut = k[k > t]
            n[i] = ((fut <= t + W)).sum()
            if len(fut):
                dte[i] = (fut.min() - t) / np.timedelta64(1, 'D')
        res[f'cal_n_{ty}'], res[f'cal_dte_{ty}'] = n, dte
    return pd.DataFrame(res, index=pd.DatetimeIndex(ud))


def _roll_series(D, H):
    """Д7.1: агрегаты по датам <= t: среднее H, изменение за H, СКО дневных изменений за H, отношение средних 5/22."""
    out = {}
    for c in D.columns:
        s = D[c].astype(float).ffill()
        pos = bool((s.dropna() > 0).all())
        ch = np.log(s).diff() if pos else s.diff()
        out[f'agg{H}_mean_{c}'] = s.rolling(H, min_periods=max(2, H // 2)).mean()
        out[f'agg{H}_chg_{c}'] = (np.log(s) - np.log(s.shift(H))) if pos else (s - s.shift(H))
        out[f'agg{H}_std_{c}'] = ch.rolling(H, min_periods=max(2, H // 2)).std()
        out[f'agg_wm_{c}'] = s.rolling(5, min_periods=3).mean() / s.rolling(22, min_periods=11).mean()
    return pd.DataFrame(out, index=D.index)


def agg_sources():
    daily = [f'close_{x}' for x in MOEX_IDX] + [f'rv_{x}' for x in MOEX_IDX] + list(YAHOO) + ['si_fut']
    drop = daily + [f'idx_{x}_rv_annualized_{s}' for x in MOEX_IDX for s in ('lag1', 'ret1')] \
        + [f'yahoo_{x}_{s}' for x in YAHOO for s in ('lag1', 'ret1')] + ['si_fut_lag1', 'si_fut_diff1']
    return daily, drop


def f_agg(A, h):
    daily, _ = agg_sources()
    H = 22 if h >= 22 else 5
    return _roll_series(by_date(A, daily), H)


def f_pc1(A, min_obs=250):
    """Д7.2: PC1 стандартизованных рядов волатильности глобальных рынков; среднее, СКО, ковариация и нагрузки — по датам <= t."""
    D = by_date(A, GLOBAL_VOL).astype(float).ffill()
    X = D.values
    ok = ~np.isnan(X).any(1)
    pc = np.full(len(D), np.nan)
    S1, S2, n = np.zeros(X.shape[1]), np.zeros((X.shape[1], X.shape[1])), 0
    for i in range(len(D)):
        if ok[i]:
            S1 += X[i]; S2 += np.outer(X[i], X[i]); n += 1
        if n >= min_obs and ok[i]:
            mu = S1 / n
            cov = S2 / n - np.outer(mu, mu)
            sd = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
            cor = cov / np.outer(sd, sd)
            w, v = np.linalg.eigh(cor)
            v1 = v[:, -1] * np.sign(v[:, -1].sum() or 1)
            pc[i] = ((X[i] - mu) / sd) @ v1
    s = pd.Series(pc, index=D.index)
    return pd.DataFrame({'pc1_gvol': s, 'pc1_gvol_chg22': s - s.shift(22)}, index=D.index)


def rvi_1850(path=RVI10):
    """RVI на 18:50 дня t: закрытие последней 10-минутной свечи, закончившейся до 18:50 (begin < 18:50)."""
    c = pd.read_parquet(path)
    b = pd.to_datetime(c['begin'])
    c = c[(b.dt.hour * 60 + b.dt.minute) < 18 * 60 + 50].copy()
    c['d'] = pd.to_datetime(c['begin']).dt.normalize()
    return c.sort_values('begin').groupby('d')['close'].last()


def f_rvi(A, s1850=None):
    """Д7.4: уровень RVI 18:50, перцентиль за 252 наблюдения (<= t), отношение к 22-дневной RV IMOEX (уже с лагом доступности)."""
    s = rvi_1850() if s1850 is None else s1850
    pct = s.rolling(252, min_periods=60).apply(lambda x: (x <= x[-1]).mean(), raw=True)
    R = pd.DataFrame({'rvi1850': s, 'rvi1850_pct252': pct})
    ud = pd.DatetimeIndex(sorted(pd.unique(pd.to_datetime(A['date']))))
    R = pd.merge_asof(pd.DataFrame(index=ud).rename_axis('date').reset_index(), R.rename_axis('date').reset_index(), on='date').set_index('date')
    imx = by_date(A, ['idx_IMOEX_rv_annualized_ma22'])['idx_IMOEX_rv_annualized_ma22'].reindex(ud)
    R['rvi1850_ratio_imoex22'] = R['rvi1850'] / 100 / imx
    return R


def f_hadd(A, h):
    """Д6, добавки по горизонтам (объявлены владельцем до B)."""
    g = A.groupby('ticker')
    out = {}
    if h >= 22:
        out['rv_mean44'] = g['rv'].transform(lambda s: s.rolling(44, min_periods=22).mean())
        out['rv_mean66'] = g['rv'].transform(lambda s: s.rolling(66, min_periods=33).mean())
        out['rv_w_over_m'] = A['rv_w'] / A['rv_m']
        out['rv_std22'] = g['rv'].transform(lambda s: s.rolling(22, min_periods=11).std())
        D = by_date(A, ['rv_IMOEX'] + list(MACRO)).astype(float).ffill()
        M = D.rolling(22, min_periods=11).mean().add_prefix('m22_')
        for c in M.columns:
            out[c] = A['date'].map(M[c])
    else:
        D = by_date(A, ['close_RVI']).astype(float).ffill()
        out['rvi_chg5'] = A['date'].map(D['close_RVI'] - D['close_RVI'].shift(5))
        out['range5'] = g['close'].transform(lambda s: np.log(s.rolling(5, min_periods=3).max() / s.rolling(5, min_periods=3).min()))
    return pd.DataFrame(out, index=A.index)


CANDLES = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/moex_discovery/data/final_v6/candles_10m')
CLOSE1850 = L2DIR / 'close1850.parquet'


def close_1850():
    """Цена закрытия на 18:50 дня t (последняя 10-минутная свеча с началом < 18:50) для 17 акций и IMOEX; кэш в data/regen/L2/close1850.parquet."""
    if CLOSE1850.exists():
        return pd.read_parquet(CLOSE1850)
    out = []
    for f in sorted((CANDLES / 'stocks').glob('*.parquet')) + [CANDLES / 'external/IMOEX.parquet']:
        c = pd.read_parquet(f)
        ts = pd.to_datetime(c['timestamp'] if 'timestamp' in c else c['begin'])
        c = c.assign(ts=ts)[(ts.dt.hour * 60 + ts.dt.minute) < 18 * 60 + 50]
        g = c.sort_values('ts').groupby(c['ts'].dt.normalize())['close'].last()
        out.append(pd.DataFrame({'date': g.index, 'ticker': f.stem, 'close1850': g.values}))
    r = pd.concat(out, ignore_index=True)
    L2DIR.mkdir(parents=True, exist_ok=True); r.to_parquet(CLOSE1850, index=False)
    return r


def f_state(A, c1850=None):
    """Д10.1 состояние рынка (по датам <= t): просадка IMOEX от 252-дневного максимума, доходности IMOEX 22/66 (закрытие 18:50), средняя попарная
    корреляция доходностей 17 акций за 22 дня, кросс-секционная дисперсия доходностей дня t, тренд Brent 66 (ряд с лагом доступности из матрицы).
    RV RGBI / CNY-RUB / Brent за 22 дня уже есть в матрице (idx_RGBI_*_ma22, idx_CNYRUB_TOM_*_ma22, yahoo_brent_vol22) — не дублируются."""
    c = close_1850() if c1850 is None else c1850
    P = c.pivot(index='date', columns='ticker', values='close1850').sort_index()
    ud = pd.DatetimeIndex(sorted(pd.unique(pd.to_datetime(A['date']))))
    P = P.reindex(P.index.union(ud)).ffill(limit=5)
    R = np.log(P).diff()
    R = R.where(R.abs() < 0.4)  # корпоративные действия (консолидации) -> пропуск
    imx = P['IMOEX']
    st = R.drop(columns='IMOEX')
    out = pd.DataFrame(index=P.index)
    out['imoex_dd252'] = imx / imx.rolling(252, min_periods=120).max() - 1
    out['imoex_ret22'] = np.log(imx / imx.shift(22))
    out['imoex_ret66'] = np.log(imx / imx.shift(66))
    out['xs_disp_ret'] = st.std(axis=1)
    vals = st.values
    pc = np.full(len(st), np.nan)
    for i in range(21, len(st)):
        w = vals[i - 21:i + 1]
        ok = (~np.isnan(w)).sum(0) >= 15
        if ok.sum() >= 5:
            C = pd.DataFrame(w[:, ok]).corr().values
            pc[i] = np.nanmean(C[np.triu_indices_from(C, 1)])
    out['avg_pair_corr22'] = pc
    b = by_date(A, ['brent'])['brent'].astype(float).ffill()
    out = out.reindex(ud)
    out['brent_trend66'] = np.log(b / b.shift(66)).reindex(ud)
    return out


def markov_filtered(A, min_obs=250):
    """Д10.2: фильтрованная (не сглаженная) вероятность режима высокой волатильности двухрежимной MarkovRegression на ln RV тикера
    (переключаются среднее и дисперсия). Параметры — по данным до конца года Y-1 (расширяющееся окно, переоценка раз в год = шаг СВ);
    фильтр по ряду до конца года Y с этими параметрами -> вероятности дат года Y (фильтр причинный)."""
    import warnings as _w
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    res = []
    for tk, g in A[['date', 'ticker', 'rv', 'year']].sort_values('date').groupby('ticker'):
        y = np.log(g['rv'].clip(lower=1e-10).values)
        yrs = g['year'].values
        for Y in sorted(set(yrs)):
            fitm = yrs < Y
            if fitm.sum() < min_obs:
                continue
            with _w.catch_warnings():
                _w.simplefilter('ignore')
                m0 = MarkovRegression(y[fitm], k_regimes=2, trend='c', switching_variance=True)
                r0 = None
                for sd in (None, 42, 43, 44, 45):  # детерминизм: сначала без случайных стартов, затем фиксированные seed для search_reps
                    try:
                        if sd is None:
                            r0 = m0.fit(disp=False)
                        else:
                            np.random.seed(sd); r0 = m0.fit(disp=False, search_reps=5)
                        if np.all(np.isfinite(r0.params)):
                            break
                    except Exception:
                        r0 = None
                if r0 is None:
                    continue
                hi = int(np.argmax(r0.params[[i for i, n in enumerate(m0.param_names) if n.startswith('const')]]))
                upto = yrs <= Y
                f = MarkovRegression(y[upto], k_regimes=2, trend='c', switching_variance=True).filter(r0.params)
                pr = np.asarray(f.filtered_marginal_probabilities)[:, hi]
            sel = yrs[upto] == Y
            res.append(pd.DataFrame({'date': g['date'].values[upto][sel], 'ticker': tk, 'ms_p_high': pr[sel]}))
    return pd.concat(res, ignore_index=True)


MS_CACHE = L2DIR / 'markov_filtered.parquet'


def f_regime(A, ms=None, s1850=None):
    """Д10.2 режим волатильности (по датам <= t): режим по перцентилю RV_m тикера (терцили, расширяющееся окно) и дней в текущем режиме,
    дней с последнего скачка и число скачков за 22/66 (скачок: доля J/RV >= 0,3), RVI(18:50) − 100·RV_IMOEX_22, вероятность MarkovRegression."""
    out = pd.DataFrame(index=A.index)
    g = A.groupby('ticker')
    pct = g['rv_m'].transform(lambda s: s.expanding(min_periods=60).rank(pct=True))
    reg = np.where(pct.isna(), np.nan, np.floor(np.clip(pct, 0, 0.9999) * 3))
    out['rvm_regime'] = reg
    tmp = pd.DataFrame({'t': A['ticker'], 'r': reg, 'j': ((A['jump'] / A['rv']) >= 0.3).astype(float)})
    def run_len(s):
        chg = s.ne(s.shift()).cumsum()
        return s.groupby(chg).cumcount() + 1
    out['rvm_regime_days'] = tmp.groupby('t')['r'].transform(run_len).where(~np.isnan(reg))
    def since(s):
        idx = np.arange(len(s)); last = pd.Series(np.where(s.values > 0, idx, np.nan), index=s.index).ffill()
        return pd.Series(idx, index=s.index) - last
    out['days_since_jump'] = tmp.groupby('t')['j'].transform(since)
    out['n_jumps22'] = tmp.groupby('t')['j'].transform(lambda s: s.rolling(22, min_periods=11).sum())
    out['n_jumps66'] = tmp.groupby('t')['j'].transform(lambda s: s.rolling(66, min_periods=33).sum())
    R = f_rvi(A, s1850)
    out['rvi_minus_rvimoex22'] = A['date'].map(R['rvi1850'] - 100 * by_date(A, ['idx_IMOEX_rv_annualized_ma22'])['idx_IMOEX_rv_annualized_ma22'].reindex(R.index)).values
    if ms is None:
        ms = pd.read_parquet(MS_CACHE) if MS_CACHE.exists() else markov_filtered(A)
        if not MS_CACHE.exists():
            ms.to_parquet(MS_CACHE, index=False)
    out['ms_p_high'] = A[['date', 'ticker']].merge(ms, on=['date', 'ticker'], how='left')['ms_p_high'].values
    return out


def f_rate(A, ev=None):
    """Д10.3 цикл ставки (по датам <= t): направление последнего изменения ключевой ставки (ряд as-of по публикации из матрицы), число изменений
    за 182 дня, дней с последнего заседания ЦБ (плановые и состоявшиеся внеочередные с датой <= t), дней до планового заседания (known_from <= t),
    структурные маркеры: дней после 24.02.2022 и после 22.06.2020 (0 до даты)."""
    kr = by_date(A, ['key_rate_cbr'])['key_rate_cbr'].astype(float).ffill()
    d = kr.diff()
    nz = d.where(d != 0)
    out = pd.DataFrame(index=kr.index)
    out['kr_last_dir'] = np.sign(nz).ffill().fillna(0)
    chg = (d.fillna(0) != 0).astype(float)
    out['kr_n_chg_6m'] = chg.rolling('182D').sum()
    cb = pd.read_csv(CAL / 'cbr.csv')
    held = np.array(sorted(pd.to_datetime(cb['date']).values))  # состоявшиеся заседания (внеочередное известно в день заседания)
    ix = out.index.values
    pos = np.searchsorted(held, ix, side='right') - 1
    out['days_since_cbr'] = np.where(pos >= 0, (ix - held[np.clip(pos, 0, None)]) / np.timedelta64(1, 'D'), np.nan)
    ev = ev or load_calendar(('cbr',))
    out['days_to_cbr_plan'] = f_cal(out.index, 22, ev)['cal_dte_cbr'].values
    out['days_after_20220224'] = np.clip((ix - np.datetime64('2022-02-24')) / np.timedelta64(1, 'D'), 0, None)
    out['days_after_20200622'] = np.clip((ix - np.datetime64('2020-06-22')) / np.timedelta64(1, 'D'), 0, None)
    return out


def f_season(A, ev=None):
    """Д10.4 сезонность: дивидендный сезон (май–июль), дней до конца квартала, неделя экспирации (по правилу, known_from <= t), месяц."""
    ud = pd.DatetimeIndex(sorted(pd.unique(pd.to_datetime(A['date']))))
    out = pd.DataFrame(index=ud)
    out['div_season'] = ud.month.isin([5, 6, 7]).astype(float)
    qe = ud.to_period('Q').end_time.normalize()
    out['days_to_qend'] = (qe - ud).days.astype(float)
    ev = ev or load_calendar(('expirations',))
    d, kf = ev['expirations']
    wk = np.zeros(len(ud))
    for i, t in enumerate(ud.values):
        k = d[kf <= t]
        wk[i] = float(np.any((k >= t - np.timedelta64(int(pd.Timestamp(t).dayofweek), 'D')) & (k <= t + np.timedelta64(6 - int(pd.Timestamp(t).dayofweek), 'D'))))
    out['expiry_week'] = wk
    out['month'] = ud.month.astype(float)
    return out


def build_features(A, FC, h, cfg):
    """A с добавленными столбцами рычагов; список признаков; категориальные; монотонные."""
    A = A.copy()
    cols = list(FC)
    extra = []
    if cfg.get('agg'):
        _, drop = agg_sources()
        cols = [c for c in cols if c not in set(drop)]
        G = f_agg(A, h)
        for c in G.columns:
            A[c] = A['date'].map(G[c]).values
        extra += list(G.columns)
    if cfg.get('cs'):
        G = f_cs(A); A[G.columns] = G; extra += list(G.columns)
    if cfg.get('cat'):
        G = f_cat(A); A[G.columns] = G; extra += list(G.columns)
    if cfg.get('cal'):
        G = f_cal(A['date'], h, load_calendar())
        for c in G.columns:
            A[c] = A['date'].map(G[c]).values
        extra += list(G.columns)
    if cfg.get('pc1'):
        G = f_pc1(A)
        for c in G.columns:
            A[c] = A['date'].map(G[c]).values
        extra += list(G.columns)
    if cfg.get('rvi'):
        G = f_rvi(A)
        for c in G.columns:
            A[c] = A['date'].map(G[c]).values
        extra += list(G.columns)
    if cfg.get('hadd'):
        G = f_hadd(A, h); A[G.columns] = G; extra += list(G.columns)
    for key, fn in (('state', f_state), ('rate', f_rate), ('season', f_season)):  # Д10.1, Д10.3, Д10.4 — ряды уровня даты
        if cfg.get(key):
            G = fn(A)
            for c in G.columns:
                A[c] = A['date'].map(G[c]).values
            extra += list(G.columns)
    if cfg.get('regime'):  # Д10.2 — уровень тикер-даты
        G = f_regime(A); A[G.columns] = G.values; extra += list(G.columns)
    cols += [c for c in extra if c not in cols]
    cats = [c for c in cols if c.startswith('cat_')]
    mono = [c for c in MONO if c in cols] if cfg.get('mono') else []
    return A, cols, cats, mono


# ---------------------------------------------------------------- HAR-J и h=1 по шагам (для Д7.5, Д8.12, Д8.6)
def harj_cols(A):
    jump = [c for c in A.columns if 'jump' in c.lower() and 'target' not in c.lower() and 'idx_' not in c.lower() and not c.startswith(('d7_', 'd10_'))]
    return [f for f in ['rv_d', 'rv_w', 'rv_m'] + [c for c in jump if c not in ('rv_d', 'rv_w', 'rv_m')] if f in A.columns]


def har_ln(tp, parts, h, hc, w=None, huber=False, y=None):
    """HAR-J (как gibrid_har.har_forecasts): МНК на ln, обучение на tp; ln-прогнозы для каждой части (tp — внутри выборки).
    w — веса наблюдений (Д11.5); huber — sklearn HuberRegressor по умолчанию (Д11.4); y — своя цель (ln; Д11.2 обратная подгонка)."""
    from sklearn.linear_model import HuberRegressor
    X = lambda d: clean(np.log(d[hc].values.clip(1e-10)))
    yy = np.log(tp[tcol(h)].values + 1e-10) if y is None else y
    m = (HuberRegressor() if huber else LinearRegression()).fit(X(tp), yy, sample_weight=w)
    return [m.predict(X(d)) for d in parts]


def har_oof(tp, h, hc, k=5, w=None, huber=False):
    """Д11.1: ln HAR-J вне свёртки на обучающем окне — k фолдов по времени (по датам подряд), точка фолда — от HAR-J без её фолда."""
    d = pd.to_datetime(tp['date']).values
    ud = np.array(sorted(set(d)))
    edges = np.array_split(ud, k)
    out = np.zeros(len(tp))
    for e in edges:
        m = np.isin(d, e)
        out[m] = har_ln(tp[~m], [tp[m]], h, hc, None if w is None else w[~m], huber)[0]
    return out


RQ_CACHE = L2DIR / 'rq1850.parquet'
_RQ = {}


def rq_1850():
    """Д11.5: реализованная квартичность дня t по 10-минутным барам окна [09:50; 18:50): RQ = (M/3)·Σ r⁴ (r — доходности закрытие-закрытие
    соседних баров окна, M — их число). Бары — candles_10m/stocks (карантинные бары не исключались — данных о карантине в этих файлах нет)."""
    if os.environ.get('L2_RQ') == 'v3_28':  # Д12.4: RQ дня t из features_pit_v3_28 (без баров карантина, PR #36)
        if '_rq28' not in _RQ:
            _RQ['_rq28'] = pd.read_parquet(REGEN / 'features_pit_v3_28.parquet', columns=['date', 'ticker', 'rq'])
        return _RQ['_rq28']
    if RQ_CACHE.exists():
        return pd.read_parquet(RQ_CACHE)
    out = []
    for f in sorted((CANDLES / 'stocks').glob('*.parquet')):
        c = pd.read_parquet(f)
        ts = pd.to_datetime(c['timestamp'])
        mins = ts.dt.hour * 60 + ts.dt.minute
        c = c.assign(ts=ts, d=ts.dt.normalize())[(mins >= 9 * 60 + 50) & (mins < 18 * 60 + 50)].sort_values('ts')
        c['r'] = np.log(c['close']).groupby(c['d']).diff()
        g = c.dropna(subset=['r']).groupby('d')['r']
        rq = (g.count() / 3) * g.apply(lambda r: float((r ** 4).sum()))
        out.append(pd.DataFrame({'date': rq.index, 'ticker': f.stem, 'rq': rq.values}))
    r = pd.concat(out, ignore_index=True)
    r.to_parquet(RQ_CACHE, index=False)
    return r


def rq_weights(tp):
    """1/RQ_t, обрезка по 1/99 перцентилям окна, нормировка к среднему 1 (строго по дню t строки)."""
    rq = tp[['date', 'ticker']].merge(rq_1850(), on=['date', 'ticker'], how='left')['rq'].values
    w = 1.0 / np.where(rq > 0, rq, np.nan)
    w = np.where(np.isnan(w), np.nanmedian(w), w)
    lo, hi = np.percentile(w, [1, 99])
    w = np.clip(w, lo, hi)
    return w / w.mean()


def split(A, h, step):
    ty = TEST_YEARS[step - 1]; t = tcol(h); tr = A[A.year < ty]; vy = sorted(tr.year.unique())[-1]
    return tr[tr.year != vy].dropna(subset=[t]), tr[tr.year == vy].dropna(subset=[t]), A[A.year == ty].dropna(subset=[t]), vy, ty


def excluded(model, h, tag):
    p = REGEN / f'{tag}_excluded_features.csv'
    if not p.exists():
        return set()
    e = pd.read_csv(p)
    return set(e[(e.model == model) & (e.h == h)].feature)


def step_data(A, cols, h, step, model, cfg, base_tag, h1=None, half=False, ex_step=None):
    """Массивы шага: обучение (все строки окна; half=True — 50% страт. тикер x год, seed 42), val, test; смещения и веса.
    ex_step — шаг, по которому применяется исключение признаков B (для бинов окна шага 1, передаваемых на шаги >= 2)."""
    tp, vi, td, vy, ty = split(A, h, step)
    m17 = lambda d: d[d['ticker'].isin(set(SECTOR))]  # Д12.1: 17 основных тикеров
    if cfg.get('train28'):  # Д12.1: бустинг обучается на 28 тикерах (матрица features_pit_v3_28), val/test — только 17
        vi, td = m17(vi), m17(td)
        if cfg.get('drop_tk'):  # справочно: 26 тикеров без слабых (UTAR, RGSS)
            tp = tp[~tp['ticker'].isin(set(cfg['drop_tk']))]
    ex = excluded(model, h, base_tag) if (ex_step or step) > 1 else set()
    ex |= set(cfg.get('drop', {}).get(model, []))
    keep = [c for c in cols if c not in ex]
    if half:
        tp = tp.groupby(['ticker', 'year'], group_keys=False).sample(frac=.5, random_state=42)
    parts = {'tr': tp, 'va': vi, 'te': td}
    X = {k: d[keep].values.astype(np.float64) for k, d in parts.items()}
    names = list(keep)
    if cfg.get('harf') or cfg.get('v9'):
        hc = harj_cols(A)
        tp_full = split(A, h, step)[0]
        if cfg.get('train28') and not cfg.get('har28'):  # Д12.1: HAR-J — на 17 (har28 — справочно на 28)
            tp_full = m17(tp_full)
        wh = rq_weights(tp_full) if cfg.get('rqw') else None
        hl = dict(zip(parts, har_ln(tp_full, list(parts.values()), h, hc, wh, bool(cfg.get('huber')))))
        if cfg.get('har_oof') and not half:  # Д11.1
            hl['tr'] = har_oof(tp_full, h, hc, 5, wh, bool(cfg.get('huber')))
    else:
        hl = None
    if cfg.get('harf'):
        for k in X:
            X[k] = np.column_stack([X[k], hl[k]])
        names.append('har_ln')
    if cfg.get('h1f'):
        for k, d in parts.items():
            v = d[['date', 'ticker']].merge(h1[step], on=['date', 'ticker'], how='left')['h1_ln'].values
            X[k] = np.column_stack([X[k], v])
        names.append('h1_ln')
    for k in X:
        X[k] = clean(X[k]).astype(np.float32)
    y = {k: d[tcol(h)].values for k, d in parts.items()}
    off = {k: (hl[k] if cfg.get('v9') else np.zeros(len(parts[k]))) for k in parts}
    if cfg.get('rel'):  # Д9.1: цель ln(RV_avg(t+1..t+h) / RV_m(t)); прогноз восстанавливается умножением на RV_m(t)
        off = {k: off[k] + np.log(np.clip(parts[k]['rv_m'].values, 1e-10, None)) for k in parts}
    w = rq_weights(tp) if cfg.get('rqw') else None  # Д11.5
    if cfg.get('hl'):
        age = (tp['date'].max() - pd.to_datetime(tp['date'])).dt.days.values / 365.25
        w = 0.5 ** (age / float(cfg['hl']))
    ytr_ln = np.log(y['tr'] + 1e-10)
    if cfg.get('wins'):
        q = tp.assign(_y=ytr_ln).groupby('ticker')['_y'].quantile([.01, .99]).unstack()
        lo, hi = tp['ticker'].map(q[0.01]).values, tp['ticker'].map(q[0.99]).values
        ytr_ln = np.clip(ytr_ln, lo, hi)
    return dict(parts=parts, X=X, y=y, ytr_ln=ytr_ln, off=off, w=w, names=names, vy=vy, ty=ty,
                cats=[i for i, c in enumerate(names) if c.startswith('cat_')],
                mono=[1 if c in set(MONO) and cfg.get('mono') else 0 for c in names])


# ---------------------------------------------------------------- обучение одной точки
def _ref_xgb(D, max_bin):
    ft = ['c' if i in D['cats'] else 'q' for i in range(len(D['names']))]
    return ft


def make_ref(model, A, cols, h, cfg, base_tag, h1=None, max_bin=MAX_BIN):
    """Бины по всем строкам обучающего окна шага 1 (признаки как на шагах >= 2) — reference для шагов 2-6 (как tune_Bv2.py)."""
    D = step_data(A, cols, h, 1, model, cfg, base_tag, h1=h1, ex_step=2)
    y = D['y']['tr'] if cfg.get('obj') == 'gamma' else D['ytr_ln']
    if model == 'xgb':
        return xgb.QuantileDMatrix(D['X']['tr'], y, max_bin=64 if max_bin == 63 else max_bin, feature_types=_ref_xgb(D, max_bin), enable_categorical=bool(D['cats']))
    dp = {'max_bin': max_bin, 'feature_pre_filter': False, 'verbosity': -1}
    if cfg.get('linear_lambda') is not None:
        dp['linear_tree'] = True
    return lgb.Dataset(D['X']['tr'], y, params=dp,
                       categorical_feature=D['cats'] or 'auto', free_raw_data=False)


def fit(model, p, D, cfg, seed=42, nmax=NMAX, ref=None, max_bin=MAX_BIN, prune=None):
    xbin = 64 if max_bin == 63 else max_bin
    """-> (ln-прогноз val, ln-прогноз test, best_iteration, gain по D['names'], модель). ES — нативная метрика на val-годе, patience 30.
    ref — объект бинов (Dataset/QuantileDMatrix) окна шага 1 для шагов >= 2 (как tune_Bv2.py); prune — (итерация, порог QLIKE)."""
    nmax = int(cfg.get('nmax', nmax))  # Д9.3: потолок 800
    gam = cfg.get('obj') == 'gamma'
    bm = gam and bool(cfg.get('v9'))  # смещение как base_margin/init_score — только gamma + V9 (mse: метка = ln y - смещение)
    Xtr, Xv, Xt = D['X']['tr'], D['X']['va'], D['X']['te']
    if gam:
        ytr, yv = D['y']['tr'], D['y']['va']
    else:
        ytr, yv = D['ytr_ln'] - D['off']['tr'], np.log(D['y']['va'] + 1e-10) - D['off']['va']
    if model == 'xgb':
        ft = _ref_xgb(D, max_bin)
        kw = dict(feature_types=ft, enable_categorical=bool(D['cats']))
        dtr = xgb.QuantileDMatrix(Xtr, ytr, weight=D['w'], base_margin=D['off']['tr'] if bm else None, max_bin=xbin, ref=ref, **kw)
        dva = xgb.QuantileDMatrix(Xv, yv, base_margin=D['off']['va'] if bm else None, ref=dtr, max_bin=xbin, **kw)
        prm = dict(tree_method='hist', max_bin=xbin, max_depth=int(p['max_depth']), eta=p['learning_rate'], subsample=p['subsample'],
                   colsample_bytree=p['colsample_bytree'], min_child_weight=p['min_child_weight'], alpha=p['reg_alpha'], reg_lambda=p['reg_lambda'],
                   nthread=1, seed=seed, verbosity=0)
        if gam:
            prm.update(objective='reg:gamma', eval_metric='gamma-nloglik')
            if not cfg.get('v9'):
                prm['base_score'] = float(np.mean(ytr))
        elif cfg.get('huber'):  # Д11.4
            prm.update(objective='reg:pseudohubererror', eval_metric='mphe')
        else:
            prm.update(objective='reg:squarederror', eval_metric='rmse')
        if any(D['mono']):
            prm['monotone_constraints'] = '(' + ','.join(map(str, D['mono'])) + ')'
        cbs = []
        if prune:
            cbs.append(_XPrune(prune, D, bm))
        b = xgb.train(prm, dtr, nmax, evals=[(dva, 'val')], early_stopping_rounds=PAT, verbose_eval=False, callbacks=cbs)
        pruned = bool(cbs and cbs[0].pruned)
        bi = prune[0] - 1 if pruned else int(b.best_iteration)
        ir = (0, bi + 1)
        g = b[:bi + 1].get_score(importance_type='total_gain')
        gain = np.array([g.get(f'f{i}', 0.0) for i in range(Xtr.shape[1])])
        pv = b.predict(xgb.DMatrix(Xv, feature_types=ft, enable_categorical=bool(D['cats']), base_margin=D['off']['va'] if bm else None), iteration_range=ir, output_margin=True)
        pt = b.predict(xgb.DMatrix(Xt, feature_types=ft, enable_categorical=bool(D['cats']), base_margin=D['off']['te'] if bm else None), iteration_range=ir, output_margin=True)
        if not bm:
            pv, pt = pv + D['off']['va'], pt + D['off']['te']
        return pv, pt, bi, gain, pruned
    dprm = {'max_bin': max_bin, 'feature_pre_filter': False, 'verbosity': -1}
    if cfg.get('linear_lambda') is not None:  # Д9.2 (только LightGBM)
        dprm['linear_tree'] = True
    dtr = lgb.Dataset(Xtr, ytr, weight=D['w'], init_score=D['off']['tr'] if gam and cfg.get('v9') else None, reference=ref, params=dprm,
                      categorical_feature=D['cats'] or 'auto', free_raw_data=False)
    dva = lgb.Dataset(Xv, yv, init_score=D['off']['va'] if gam and cfg.get('v9') else None, reference=dtr, categorical_feature=D['cats'] or 'auto', free_raw_data=False)
    prm = dict(objective='gamma' if gam else ('huber' if cfg.get('huber') else 'regression'), metric='gamma' if gam else ('huber' if cfg.get('huber') else 'l2'),
               boosting_type='gbdt', max_depth=-1,
               num_leaves=int(p['num_leaves']), learning_rate=p['learning_rate'], bagging_fraction=p['bagging_fraction'], bagging_freq=1,
               feature_fraction=p['feature_fraction'], min_data_in_leaf=int(p['min_data_in_leaf']), reg_alpha=p['reg_alpha'], reg_lambda=p['reg_lambda'],
               max_bin=max_bin, verbosity=-1, seed=seed, num_threads=1, force_col_wise=True)
    if cfg.get('quantile') is not None:  # Д12.6: квантильный остаточный ученик (LightGBM; цель — ln RV − ln HAR-J, V9), ES по pinball-потере val-года
        prm.update(objective='quantile', alpha=float(cfg['quantile']), metric='quantile')
    if any(D['mono']):
        prm['monotone_constraints'] = D['mono']
    if p.get('extra_trees') in (True, 1, 'True', 1.0):  # Д9.4
        prm['extra_trees'] = True
    if cfg.get('linear_lambda') is not None:
        prm.update(linear_tree=True, linear_lambda=float(cfg['linear_lambda']))
    pr = []
    cbs = [lgb.early_stopping(PAT, verbose=False)]
    if prune:
        it, thr = prune

        def _pr(env):
            if env.iteration == it - 1 and _q_ln(D, env.model.predict(Xv, raw_score=True), gam) > thr:
                pr.append(1); raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)
        cbs.insert(0, _pr)
    b = lgb.train(prm, dtr, nmax, valid_sets=[dva], callbacks=cbs)
    pv, pt = b.predict(Xv, raw_score=True), b.predict(Xt, raw_score=True)
    pv, pt = pv + D['off']['va'], pt + D['off']['te']  # mse: метка = ln y - смещение; gamma+v9: init_score не входит в raw_score
    return pv, pt, int(b.best_iteration), b.feature_importance('gain'), bool(pr)


def _q_ln(D, raw, gam):
    """QLIKE на val по сырому выходу (для правила снятия на итерации 100)."""
    return qlike(D['y']['va'], np.exp(raw + D['off']['va']))


class _XPrune(xgb.callback.TrainingCallback):
    def __init__(self, prune, D, gam):
        self.it, self.thr, self.D, self.gam, self.pruned = prune[0], prune[1], D, gam, False
        self.dm = xgb.DMatrix(D['X']['va'], feature_types=['c' if i in D['cats'] else 'q' for i in range(len(D['names']))],
                              enable_categorical=bool(D['cats']), base_margin=D['off']['va'] if gam else None)

    def after_iteration(self, model, epoch, evals_log):
        if epoch == self.it - 1:
            raw = model.predict(self.dm, output_margin=True)  # gamma: base_margin уже внутри -> вычесть (смещение добавит _q_ln)
            if _q_ln(self.D, raw - self.D['off']['va'] if self.gam else raw, self.gam) > self.thr:
                self.pruned = True
        return self.pruned


# ---------------------------------------------------------------- оценка: Б0.1 вне выборки (как gibrid_bias.py), шаги 2-6
CODES = ('ord', 'rob', 'sm', 'smrob')


def s2_est(code, E, st):
    e = np.concatenate([E[j] for j in range(max(1, st - 2) if code in ('sm', 'smrob') else st, st + 1) if j in E])
    if code in ('rob', 'smrob'):
        return float((1.4826 * np.median(np.abs(e - np.median(e)))) ** 2)
    return float(np.var(e))


def evaluate(val, lo=2, hi=6):
    """val: DataFrame[step, rv_actual, rv_pred] шагов 1..6 одной модели -> {код: (среднее шагов lo..hi, [по шагам])}; 'raw' — без поправки.
    Поправка на val шага k — оценка s2 по остаткам val-лет до шага k (шаг k-1; sm/smrob — до трёх последних)."""
    v = val[val.rv_actual > 1e-12]
    G = {st: g for st, g in v.groupby('step')}
    E = {st: np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for st, g in G.items()}
    out = {}
    for code in ('raw',) + CODES:
        qs = []
        for st in range(lo, hi + 1):
            g = G[st]
            F = 1.0 if code == 'raw' else np.exp(s2_est(code, E, st - 1) / 2)
            qs.append(qlike(g.rv_actual.values, g.rv_pred.values * F))
        out[code] = (float(np.mean(qs)), qs)
    return out


def summarize(vals, gam=False):
    """vals: {model: val DataFrame}. Основная метрика: лучший бустинг с поправкой ord (gamma — без поправки, Д1); вторая — лучший из 4 оценок."""
    ev = {m: evaluate(d) for m, d in vals.items()}
    main_code = 'raw' if gam else 'ord'
    best_m = min(ev, key=lambda m: ev[m][main_code][0])
    b4 = min(((m, c) for m in ev for c in (('raw',) if gam else CODES)), key=lambda mc: ev[mc[0]][mc[1]][0])
    return dict(metric=ev[best_m][main_code][0], best_model=best_m, best4=ev[b4[0]][b4[1]][0], best4_mc=f'{b4[0]}/{b4[1]}',
                **{f'{m}_{c}': ev[m][c][0] for m in ev for c in ev[m]}, per_step={m: ev[m][main_code][1] for m in ev})


def selected_params(tag, h):
    """tag — префикс файлов B в data/regen ({tag}_selected_h{h}.csv) либо путь к CSV выбранных точек (Д9.3: data/regen/L2/h{h}/*_selected.csv)."""
    s = pd.read_csv(tag if str(tag).endswith('.csv') else REGEN / f'{tag}_selected_h{h}.csv')
    return {(r.model, int(r.step)): r._asdict() for r in s.itertuples(index=False)}
