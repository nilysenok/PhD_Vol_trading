"""Д13 (второй круг, решение владельца 25.10, 20:55): признаки и данные рычагов поверх l2_lib на матрице features_pit_v3 (as-of).
Новые признаки (все — из данных <= 18:50 дня t; тест «будущее не влияет» — tests/test_d13_features.py):
  Д13.7 f_intraday — по 10-минутным барам data/iss_check/bars_10m (окно [09:50; 18:50) по началу бара, без карантина):
        rv_lasthour (бары с началом в [17:50; 18:50)), rv_half2 (начало >= 14:20 — вторая половина окна 9 ч), gap_on = ln(open_t / close_{t-1})
        (open — первый бар окна дня t, close_{t-1} — последний бар дня t-1 тикера, любое время), jshare = max(RV - BV, 0) / RV окна дня t.
  Д13.1 h1_cascade — ln-прогноз модели h=1 (XGBoost, точка Bv2b шага, cfg {} на v3, число деревьев = best_iteration + 1):
        строки года Y <= val-года шага — модель на годах < Y (фолды по времени вперёд; первый год окна — пусто); тестовый год — модель на окне шага.
Кросс-секционные (Д13.3), категориальные (Д13.4), монотонность (Д13.5), веса по давности (Д13.6) — функции l2_lib (f_cs, f_cat, MONO, hl)."""
import os
import numpy as np, pandas as pd
import l2_lib as L

L.FEAT = L.REGEN / 'features_pit_v3.parquet'  # конфигурация признаков — features_pit_v3.config.json
BARS = L.DATA / 'iss_check' / 'bars_10m'
D13DIR = L.REGEN / 'D13'
INTRA_CACHE = D13DIR / 'intraday1850.parquet'
CASC_CACHE = D13DIR / 'h1_cascade.parquet'


def load(h, blocks=''):
    """v3 (+ блок Д7 v3 для h=5: features_pit_v3_d7_h5.parquet)."""
    m = L.REGEN / f'features_pit_v3_d7_h{h}.parquet' if 'd7' in blocks else L.FEAT
    return L.load_base(m, (), h)


def intraday_one(c, tk):
    ts = pd.to_datetime(c['begin'])
    mins = ts.dt.hour * 60 + ts.dt.minute
    c = c.assign(ts=ts, d=ts.dt.normalize(), mins=mins).sort_values('ts')
    last_close = c.groupby('d')['close'].last()  # весь день (для close_{t-1})
    w = c[(c.mins >= 9 * 60 + 50) & (c.mins < 18 * 60 + 50)].copy()
    w['r'] = np.log(w['close']).groupby(w['d']).diff()
    w['ar1'] = w['r'].abs().groupby(w['d']).shift(1)
    g = w.groupby('d')
    rv = g['r'].apply(lambda r: float(np.nansum(r ** 2)))
    bv = (np.pi / 2) * g.apply(lambda x: float(np.nansum(x['r'].abs() * x['ar1'])))
    lh = w[w.mins >= 17 * 60 + 50].groupby('d')['r'].apply(lambda r: float(np.nansum(r ** 2)))
    h2 = w[w.mins >= 14 * 60 + 20].groupby('d')['r'].apply(lambda r: float(np.nansum(r ** 2)))
    op = g['open'].first()
    prev = last_close.shift(1).reindex(op.index)  # предыдущий торговый день тикера
    out = pd.DataFrame({'rv_lasthour': lh.reindex(rv.index), 'rv_half2': h2.reindex(rv.index),
                        'gap_on': np.log(op / prev).where(lambda s: s.abs() < 0.4).reindex(rv.index),  # |гэп| >= 0,4 — корпоративные действия -> пропуск (как f_state)
                        'jshare': (np.clip(rv - bv, 0, None) / rv.where(rv > 0)).reindex(rv.index)})
    out.index.name = 'date'
    return out.reset_index().assign(ticker=tk)


def f_intraday(bars=None, tickers=None):
    """bars: {ticker: DataFrame баров} (для теста утечек); None — читать BARS (кэш INTRA_CACHE)."""
    if bars is None and INTRA_CACHE.exists():
        return pd.read_parquet(INTRA_CACHE)
    if bars is None:
        tickers = tickers or sorted(L.SECTOR)
        bars = {tk: pd.read_parquet(BARS / f'{tk}.parquet') for tk in tickers}
    r = pd.concat([intraday_one(c, tk) for tk, c in bars.items()], ignore_index=True)
    if tickers is not None or not INTRA_CACHE.exists():
        D13DIR.mkdir(parents=True, exist_ok=True)
        if set(bars) >= set(L.SECTOR):
            r.to_parquet(INTRA_CACHE, index=False)
    return r


INTRA_COLS = ('rv_lasthour', 'rv_half2', 'gap_on', 'jshare')


def add_intraday(A, cols, I=None):
    I = f_intraday() if I is None else I
    I = I.assign(date=pd.to_datetime(I['date']))
    A = A.copy()
    key = A[['date', 'ticker']].assign(date=pd.to_datetime(A['date']))
    M = key.merge(I, on=['date', 'ticker'], how='left')
    for c in INTRA_COLS:
        A[c] = M[c].values
    return A, list(cols) + [c for c in INTRA_COLS if c not in cols]


# ---------------------------------------------------------------- Д13.1 каскад h=1 (фолды по времени вперёд)
def _h1_prm(P):
    return dict(tree_method='hist', max_bin=L.MAX_BIN, max_depth=int(P['max_depth']), eta=P['learning_rate'], subsample=P['subsample'],
                colsample_bytree=P['colsample_bytree'], min_child_weight=P['min_child_weight'], alpha=P['reg_alpha'], reg_lambda=P['reg_lambda'],
                nthread=1, seed=42, verbosity=0)


def h1_cascade_task(A1, FC1, step, Y, P, keep):
    """Прогноз h=1 строк года Y (Y <= val-год шага: модель на годах < Y; Y = тестовый год: модель на годах < val-года)."""
    import xgboost as xgb
    ty = L.TEST_YEARS[step - 1]; vy = ty - 1
    fit_upto = vy if Y == ty else Y
    tr = A1[(A1.year < fit_upto)].dropna(subset=['rv_target_h1'])
    pr = A1[A1.year == Y]
    X = lambda d: L.clean(d[keep].values).astype(np.float32)
    b = xgb.train(_h1_prm(P), xgb.DMatrix(X(tr), np.log(tr['rv_target_h1'].values + 1e-10)), int(P['best_iteration']) + 1)
    return pd.DataFrame({'date': pr.date.values, 'ticker': pr.ticker.values, 'step': step, 'h1_ln': b.predict(xgb.DMatrix(X(pr)))})


def h1_cascade_jobs(A1, FC1, steps, tag='Bv2b'):
    S = L.selected_params(tag, 1)
    jobs = []
    first = int(A1.year.min())
    for st in steps:
        P = S[('xgb', st)]
        keep = [c for c in FC1 if c not in (L.excluded('xgb', 1, tag) if st > 1 else set())]
        ty = L.TEST_YEARS[st - 1]
        for Y in list(range(first + 1, ty)) + [ty]:
            jobs.append((st, Y, P, keep))
    return jobs


def h1_dict(df):
    return {int(s): g.drop(columns='step') for s, g in df.groupby('step')}
