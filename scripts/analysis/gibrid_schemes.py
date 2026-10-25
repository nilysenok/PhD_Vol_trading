"""Схемы гибрида V1-V7 «как в коде». Источники (строки в moex_volatility/scripts/):
V1 walk_forward_full.py:258-290 (+выбор ML adaptive.py:270-273); V2 adaptive_v2.py:53-67,126-158; V3 :204-261;
V4 :309-338 (lambda=0.3, :293); V5 :394-457; V6 adaptive_v3.py:107-123; V7 adaptive_v3.py:193-223 (окно 20; Д8.3 - полураспад hl).
Веса — тройки (w_har, w_xgb, w_lgb). Смешивание (Д8.1, --blend): MODE='rv' - арифметическое на уровне RV по компонентам с их
поправками, F = sum w_k F_k (как в коде); MODE='log' - в логах, F = exp(sum w_k ln F_k) (геометрическое, те же поправки)."""
import numpy as np, pandas as pd

LAMBDA = 0.3
EQ = np.array([1 / 3, 1 / 3, 1 / 3])
MODE = 'rv'


def comb(P, W):
    """P: n x k прогнозов; W: k весов или n x k. Смешивание по MODE (rv - арифметическое, log - геометрическое)."""
    P = np.clip(np.asarray(P, dtype=float), 1e-10, None)
    if MODE == 'log':
        return np.clip(np.exp((np.log(P) * W).sum(axis=1)), 1e-10, None)
    return np.clip((P * W).sum(axis=1), 1e-10, None)


def qlike(y, p):
    p = np.clip(p, 1e-10, None)
    return float(np.mean(y / p - np.log(y / p) - 1))


def blend(w, d):
    return comb(np.column_stack([d['har'].values, d['xgb'].values, d['lgb'].values]), np.asarray(w, dtype=float))


def v1(d):
    """Лучшая ML по val-QLIKE; w_har in [0;0,60] шаг 0,01 по val-QLIKE; при w=0 — inverse-QLIKE (без клипа 0,05-0,55)."""
    y = d['rv_actual'].values
    ml = 'xgb' if qlike(y, d['xgb'].values) < qlike(y, d['lgb'].values) else 'lgb'
    m, hr = d[ml].values, d['har'].values
    P2 = np.column_stack([hr, m])
    bw, bq = 0.0, qlike(y, m)
    for w in np.arange(0, 0.61, 0.01):
        q = qlike(y, comb(P2, np.array([w, 1 - w])))
        if q < bq:
            bq, bw = q, w
    method = 'val_grid'
    if bw == 0:
        ih, im = 1.0 / (qlike(y, hr) + 1e-10), 1.0 / (qlike(y, m) + 1e-10)
        bw, method = ih / (ih + im), 'inverse_qlike'
    return (bw, 1 - bw, 0.0) if ml == 'xgb' else (bw, 0.0, 1 - bw), method, ml


def grid3(d, step=0.02):
    """adaptive_v2.py:53-67: сетка w_har in [0;0,6], w_xgb, w_lgb=остаток."""
    y, P = d['rv_actual'].values, np.column_stack([d[c].values for c in ('har', 'xgb', 'lgb')])
    bq, bw = 999, (1 / 3, 1 / 3, 1 / 3)
    for wh in np.arange(0, 0.61, step):
        for wx in np.arange(0, 1.01 - wh, step):
            wl = round(1 - wh - wx, 4)
            if wl < -0.001:
                continue
            wl = max(wl, 0)
            q = qlike(y, comb(P, np.array([wh, wx, wl])))
            if q < bq:
                bq, bw = q, (wh, wx, wl)
    return np.array(bw)


def regime(ratio):
    return 'HIGH_VOL' if ratio > 1.3 else 'LOW_VOL' if ratio < 0.7 else 'NORMAL'


def regime_adj(w, reg):
    """adaptive_v2.py:238-261: HAR-вес x1,5 (<=0,6) или x0,7 (>=0,05), остаток делится между ML пропорционально."""
    w = np.array(w, dtype=float)
    if reg == 'NORMAL':
        return w / w.sum()
    wh = min(w[0] * 1.5, 0.6) if reg == 'HIGH_VOL' else max(w[0] * 0.7, 0.05)
    rem, s = 1.0 - wh, w[1] + w[2]
    w[1:] = rem * w[1:] / s if s > 0 else rem / 2
    w[0] = wh
    return w / w.sum()


def weights_v2_v5(val1, val2, ratio):
    """val1 — валидационный год шага, val2 — год до него (или None). Возвращает {V2..V5: w}."""
    g1 = grid3(val1)
    ws = [g1] + ([grid3(val2)] if val2 is not None and len(val2) > 10 else [])
    w2 = np.mean(ws, axis=0)
    w2 = w2 / w2.sum()
    reg = regime(ratio)
    w3 = regime_adj(g1, reg)
    w4 = (1 - LAMBDA) * g1 + LAMBDA * EQ
    w5 = regime_adj(w2, reg)
    w5 = (1 - LAMBDA) * w5 + LAMBDA * EQ
    return {'V2_MultiVal': w2, 'V3_Regime': w3, 'V4_Shrinkage': w4 / w4.sum(), 'V5_Combined': w5 / w5.sum()}, reg


def v6(d):
    """adaptive_v3.py:107-123: отбросить модель с наибольшим |p-med|/med, остальным веса 1/dev (dev>=1e-8)."""
    s = np.column_stack([d['har'].values, d['xgb'].values, d['lgb'].values])
    med = np.median(s, axis=1, keepdims=True)
    dev = np.abs(s - med) / np.clip(med, 1e-10, None)
    mask = np.ones_like(s)
    mask[np.arange(len(s)), np.argmax(dev, axis=1)] = 0
    iv = 1.0 / np.clip(dev, 1e-8, None) * mask
    W = iv / iv.sum(axis=1, keepdims=True)  # эффективные веса по наблюдениям (для критерия скачков)
    return comb(s, W), W


def v7(series, window=20, lag=1, hl=None):
    """adaptive_v3.py:177-223: по тикерам, веса 1/rolling-QLIKE окна [i-window..i-1] (lag=1 как в коде), i<window — среднее.
    hl (Д8.3): вместо окна - экспоненциальное среднее потерь с полураспадом hl дней (min 10 наблюдений), тот же лаг.
    series: DataFrame date,ticker,rv_actual,har,xgb,lgb; возвращает (прогнозы, эффективные веса)."""
    s = series.sort_values(['ticker', 'date'])
    y = s['rv_actual'].values
    inv, P = [], []
    for c in ('har', 'xgb', 'lgb'):
        p = np.clip(s[c].values, 1e-10, None)
        loss = pd.Series(y / p - np.log(y / p) - 1, index=s.index)
        rq = loss.groupby(s['ticker']).transform(lambda v: (v.rolling(window) if hl is None else v.ewm(halflife=hl, min_periods=10)).mean().shift(lag))
        inv.append(1.0 / np.maximum(rq.values, 1e-6)); P.append(p)
    inv, P = np.column_stack(inv), np.column_stack(P)
    ok = ~np.isnan(inv).any(axis=1)
    w = np.where(ok[:, None], inv / np.where(ok, inv.sum(axis=1), 1)[:, None], 1 / 3)
    pred = pd.Series(comb(P, w), index=s.index).reindex(series.index)
    return pred, pd.DataFrame(w, index=s.index).reindex(series.index).values
