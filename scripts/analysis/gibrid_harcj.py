"""Д8.6: HAR-CJ (Andersen-Bollerslev-Diebold 2007) - непрерывная компонента (BV) и скачки (jump) раздельно, дневной/недельный/месячный
лаги: C_d = BV(t-1), C_w = среднее BV(t-5..t-1), C_m = среднее BV(t-22..t-1); J_d, J_w, J_m - так же по jump (как rv_d = rv(t-1) в данных).
OLS в логах ln(clip(x, 1e-10)) - та же процедура шагов, окна и обработка пропусков, что у HAR-J (gibrid_har.har_forecasts).
Также признаки состояния для гейта Д8.7: rv_d, rv_w, rv_m, VIX(t-1) = yahoo_vix_lag1, RVI(t-1) = close_RVI со сдвигом 1 день по тикеру.
Не делается здесь: «прогноз h=1 как признак бустинга на h=5/22» - это переобучение бустинга, пометка для конвейера (models/pipeline.py)."""
import gibrid_har as GH

CJ = ['c_d', 'c_w', 'c_m', 'j_d', 'j_w', 'j_m']


def add_cj(A):
    A = A.copy()
    g = A.groupby('ticker', sort=False)
    for src, p in (('bv', 'c'), ('jump', 'j')):
        lag1 = g[src].shift(1)
        A[p + '_d'] = lag1
        for k, n in (('w', 5), ('m', 22)):
            A[p + '_' + k] = lag1.groupby(A['ticker']).transform(lambda v: v.rolling(n, min_periods=1).mean())
    A['vix1'] = A['yahoo_vix_lag1']
    A['rvi1'] = g['close_RVI'].shift(1)
    return A


def forecasts(h, A):
    """A - результат add_cj. Возвращает {'val': ..., 'test': ...} с колонками date, ticker, hcj."""
    hv, ht, _ = GH.har_forecasts(h, A, CJ)
    return {p: d[['date', 'ticker', 'rv_pred']].rename(columns={'rv_pred': 'hcj'}) for p, d in (('val', hv), ('test', ht))}


def state(A):
    return A[['date', 'ticker', 'rv_d', 'rv_w', 'rv_m', 'vix1', 'rvi1']]
