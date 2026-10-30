"""Семейство S3 (Donchian) и S4 (Supertrend): трендовые сигналы daily, numba, O(T) на конфигурацию.

Семантика: SW.gen_s3_signals / gen_s4_signals / calc_supertrend @c22a4b3, сила — V4.signal_strength_s3/s4
(docs/fastbt_spec.md п. 11). Фильтры (ADX14>20, объём>SMA20 для S3, vf_pass) берутся из `ind` ядра, поэтому
искажение ADX у IRAO (скачок 20.01.2015) воспроизводится как есть. Daily-тренд-фильтра для daily нет.
ATR супертренда = SMA(TR, period): sma() повторяет pandas rolling.mean (Kahan, раздельные компенсации
сложения/вычитания), чтобы пороги сравнивались на тех же битах.
"""
from __future__ import annotations

import sys

import numpy as np
from numba import njit

from fastbt.signals.interface import WARMUP, Family, SignalSet, register

S3_GRID = {"dc_window": [10, 15, 20, 25]}
S4_GRID = {"atr_period": [10, 14, 20], "multiplier": [2.0, 2.5, 3.0, 3.5]}


@njit(cache=True)
def sma(x, w):
    """pandas Series.rolling(w, min_periods=w).mean() (NaN первые w-1 баров)."""
    n = len(x)
    out = np.full(n, np.nan)
    s = 0.0; ca = 0.0; cr = 0.0; nobs = 0; same = 0; prev = 0.0
    for i in range(n):
        if i >= w:
            v = x[i - w]
            if v == v:
                nobs -= 1
                y = -v - cr; t = s + y; cr = t - s - y; s = t
        v = x[i]
        if v == v:
            nobs += 1
            y = v - ca; t = s + y; ca = t - s - y; s = t
            same = same + 1 if v == prev else 1
            prev = v
        if nobs >= w:
            out[i] = prev if same >= nobs else s / nobs
    return out


@njit(cache=True)
def rolling_ext(x, w, is_max):
    """pandas rolling(w).max()/min() (NaN первые w-1 баров): монотонная очередь, O(T)."""
    n = len(x)
    out = np.full(n, np.nan)
    dq = np.empty(n, np.int64)
    h = 0; t = 0
    for i in range(n):
        while t > h and ((x[dq[t - 1]] <= x[i]) if is_max else (x[dq[t - 1]] >= x[i])):
            t -= 1
        dq[t] = i; t += 1
        if dq[h] <= i - w:
            h += 1
        if i >= w - 1:
            out[i] = x[dq[h]]
    return out


@njit(cache=True)
def supertrend(high, low, close, period, mult):
    """calc_supertrend c22a4b3: ST[t] (NaN до бара period). Старт — от первого бара с конечными H/L/C (прогрев относительного
    ряда не обнуляет весь ST): ST[f+period…], до него NaN, без подстановок (Д4). Начало ряда конечное (f=0) — побайтно прежнее."""
    n = len(close)
    f = 0
    while f < n and not (np.isfinite(high[f]) and np.isfinite(low[f]) and np.isfinite(close[f])):
        f += 1
    out = np.full(n, np.nan)
    if f < n:
        out[f:] = _supertrend0(high[f:], low[f:], close[f:], period, mult)
    return out


@njit(cache=True)
def _supertrend0(high, low, close, period, mult):
    """Ядро calc_supertrend c22a4b3 на ряду, начинающемся с конечного бара."""
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = sma(tr, period)
    st = np.full(n, np.nan)
    if period >= n:
        return st
    fu = np.full(n, np.nan); fl = np.full(n, np.nan); d = np.zeros(n)
    s = period
    bu = (high[s] + low[s]) / 2.0 + mult * atr[s]
    bl = (high[s] + low[s]) / 2.0 - mult * atr[s]
    fu[s] = bu; fl[s] = bl
    d[s] = 1.0 if close[s] > bu else -1.0
    st[s] = fl[s] if d[s] == 1 else fu[s]
    for t in range(s + 1, n):
        bu = (high[t] + low[t]) / 2.0 + mult * atr[t]
        bl = (high[t] + low[t]) / 2.0 - mult * atr[t]
        if np.isnan(atr[t]):
            d[t] = d[t - 1]; fu[t] = bu; fl[t] = bl; st[t] = st[t - 1]
            continue
        fu[t] = bu if (bu < fu[t - 1] or close[t - 1] > fu[t - 1]) else fu[t - 1]
        fl[t] = bl if (bl > fl[t - 1] or close[t - 1] < fl[t - 1]) else fl[t - 1]
        if d[t - 1] == 1:
            d[t] = -1.0 if close[t] < fl[t] else 1.0
        else:
            d[t] = 1.0 if close[t] > fu[t] else -1.0
        st[t] = fl[t] if d[t] == 1 else fu[t]
    return st


@njit(cache=True)
def _s3(high, low, close, adx, vs20, vol, vf, w, sig, exit_l, exit_s):
    """Вход: close пробивает Donchian[t-1] при ADX>20, vol/SMA20>1, vf_pass; выход: пробой обратной границы."""
    hc = rolling_ext(high, w, True)
    lc = rolling_ext(low, w, False)
    for i in range(1, len(close)):
        ph = hc[i - 1]; pl = lc[i - 1]
        exit_l[i] = (pl == pl) and close[i] < pl
        exit_s[i] = (ph == ph) and close[i] > ph
        if i < WARMUP or not (ph == ph):
            continue
        ok = (vs20[i] == vs20[i]) and vs20[i] > 0
        f_vc = (vol[i] / (vs20[i] if ok else 1.0)) > 1.0
        if (adx[i] == adx[i]) and adx[i] > 20 and f_vc and vf[i]:
            if close[i] > ph:
                sig[i] = 1
            elif close[i] < pl:
                sig[i] = -1


@njit(cache=True)
def _s4(close, st, adx, vf, sig, exit_l, exit_s):
    """Вход/выход: пересечение close через ST (вход при ADX>20 и vf_pass)."""
    for i in range(1, len(close)):
        if not (st[i] == st[i] and st[i - 1] == st[i - 1]):
            continue
        up = close[i] > st[i] and close[i - 1] <= st[i - 1]
        dn = close[i] < st[i] and close[i - 1] >= st[i - 1]
        exit_l[i] = close[i] < st[i] and close[i - 1] >= st[i - 1]
        exit_s[i] = close[i] > st[i] and close[i - 1] <= st[i - 1]
        if i >= WARMUP and (adx[i] == adx[i]) and adx[i] > 20 and vf[i]:
            sig[i] = 1 if up else (-1 if dn else 0)


def _alloc(k, t):
    return (np.zeros((k, t), np.int8), np.zeros((k, t), np.bool_), np.zeros((k, t), np.bool_),
            np.empty((k, t)))


def generate_s3(td, ind, grid):
    sig, el, es, stg = _alloc(len(grid), td.n)
    adx = ind["adx14"]
    row = np.minimum(np.where(np.isnan(adx), 0.0, adx) / 35.0, 1.0)
    for k, p in enumerate(grid):
        _s3(td.high, td.low, td.close, adx, ind["vol_sma20"], td.volume, ind["vf_pass"],
            int(p["dc_window"]), sig[k], el[k], es[k])
        stg[k] = row
    return SignalSet(sig, el, es, stg, list(grid), split=True)


def generate_s4(td, ind, grid):
    sig, el, es, stg = _alloc(len(grid), td.n)
    atr = ind["atr14"]
    for k, p in enumerate(grid):
        st = supertrend(td.high, td.low, td.close, int(p["atr_period"]), float(p["multiplier"]))
        _s4(td.close, st, ind["adx14"], ind["vf_pass"], sig[k], el[k], es[k])
        stg[k] = np.minimum(np.abs(td.close - st) / (1.5 * np.maximum(atr, 1e-12)), 1.0)
    return SignalSet(sig, el, es, stg, list(grid), split=True)


_LEG = {}


def _legacy_ctx(L, td, windows=(), sts=()):
    """База старого кода по тикеру (compute_base один раз) + кэши под нужную конфигурацию."""
    key = (td.ticker, td.n, id(td.close))
    if key not in _LEG:
        ind = L.sw.compute_base(td.close, td.high, td.low, td.open, td.volume)
        _LEG[key] = ind
    return _LEG[key]


def legacy_s3(L, td, p):
    ind = _legacy_ctx(L, td)
    dc = L.sw.precompute_donchian_cache(td.high, td.low, [p["dc_window"]])
    sig, (el, es) = L.sw.gen_s3_signals(td.close, td.high, td.low, ind, dc, daily_trend=None, **p)
    return sig, el, es, L.v4.signal_strength_s3(td.close, ind, dc, p)


def legacy_s4(L, td, p):
    ind = _legacy_ctx(L, td)
    stc = L.sw.precompute_supertrend_cache(td.high, td.low, td.close, [p["atr_period"]], [p["multiplier"]])
    sig, (el, es) = L.sw.gen_s4_signals(td.close, td.high, td.low, ind, stc, daily_trend=None, **p)
    return sig, el, es, L.v4.signal_strength_s4(td.close, ind, stc, p)


S3 = register(Family("S3", "Trend", S3_GRID, generate_s3, legacy_s3, module=sys.modules[__name__]))
S4 = register(Family("S4", "Trend", S4_GRID, generate_s4, legacy_s4, module=sys.modules[__name__]))
