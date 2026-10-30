"""Базовые индикаторы ядра — побитовый перенос SW.compute_base (c22a4b3, SW:204-378).

Все семейства получают этот dict. Формулы и pandas-реализации сохранены ровно,
чтобы пороговые сравнения в сигналах давали те же биты.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


def calc_sma(arr, w):
    return pd.Series(arr).rolling(w, min_periods=w).mean().values


def calc_std(arr, w):
    s = pd.Series(arr).rolling(w).std(ddof=1).values
    return np.where(np.isnan(s) | (s < 1e-12), 1e-12, s)


@njit(cache=True)
def _true_range(high, low, close):
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    return tr


def calc_atr(high, low, close, period=14):
    return pd.Series(_true_range(high, low, close)).rolling(period).mean().values


@njit(cache=True)
def _dm_tr(high, low, close):
    n = len(close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        plus_dm[i] = up if (up > dn and up > 0) else 0.0
        minus_dm[i] = dn if (dn > up and dn > 0) else 0.0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    return plus_dm, minus_dm, tr


@njit(cache=True)
def _adx_core(plus_dm, minus_dm, tr, s_tr, s_p, s_m, period):
    # начальные суммы считаются numpy (попарное суммирование) снаружи — как в SW
    n = len(tr)
    atr_s = np.full(n, np.nan)
    pdi_s = np.full(n, np.nan)
    mdi_s = np.full(n, np.nan)
    atr_s[period] = s_tr
    pdi_s[period] = s_p
    mdi_s[period] = s_m
    for i in range(period + 1, n):
        atr_s[i] = atr_s[i - 1] - atr_s[i - 1] / period + tr[i]
        pdi_s[i] = pdi_s[i - 1] - pdi_s[i - 1] / period + plus_dm[i]
        mdi_s[i] = mdi_s[i - 1] - mdi_s[i - 1] / period + minus_dm[i]
    dx = np.full(n, np.nan)
    for i in range(period, n):
        if atr_s[i] > 1e-12:
            pdi = 100.0 * pdi_s[i] / atr_s[i]
            mdi = 100.0 * mdi_s[i] / atr_s[i]
        else:
            pdi, mdi = 0.0, 0.0
        s = pdi + mdi
        dx[i] = 100.0 * abs(pdi - mdi) / s if s > 1e-12 else 0.0
    return dx


def calc_adx(high, low, close, period=14):
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    pdm, mdm, tr = _dm_tr(high, low, close)
    dx = _adx_core(pdm, mdm, tr, np.sum(tr[1:period + 1]), np.sum(pdm[1:period + 1]),
                   np.sum(mdm[1:period + 1]), period)
    adx = np.full(n, np.nan)
    start = 2 * period
    if start < n:
        adx[start] = np.nanmean(dx[period:start + 1])
        for i in range(start + 1, n):
            if not np.isnan(dx[i]) and not np.isnan(adx[i - 1]):
                adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx


def compute_base(close, high, low, open_arr, volume):
    """Базовые индикаторы. Ведущий NaN-префикс (прогрев относительного ряда) отрезается, индикаторы считаются на конечной
    части и дополняются NaN назад до исходных дат (без подстановок, Д4; vf_pass — True, как при NaN в объёме).
    Без префикса — побайтно прежнее (`_compute_base0`)."""
    close, high, low, open_arr, volume = (np.asarray(x) for x in (close, high, low, open_arr, volume))
    ok = np.isfinite(close) & np.isfinite(high) & np.isfinite(low) & np.isfinite(open_arr)
    f = int(np.argmax(ok)) if ok.any() else len(close)
    if f == 0 or f == len(close):
        return _compute_base0(close, high, low, open_arr, volume)
    part = _compute_base0(close[f:], high[f:], low[f:], open_arr[f:], volume[f:])
    out = {}
    for k, v in part.items():
        if k == "_volume":
            out[k] = volume
            continue
        full = np.full(f + len(v), True) if v.dtype == bool else np.full(f + len(v), np.nan)
        full[f:] = v
        out[k] = full
    return out


def _compute_base0(close, high, low, open_arr, volume):
    ind = {}
    n = len(close)
    ind["atr14"] = calc_atr(high, low, close, 14)
    ind["adx14"] = calc_adx(high, low, close, 14)
    ind["vol_sma20"] = calc_sma(volume, 20)
    ind["vol_sma5"] = calc_sma(volume, 5)
    atr_sma50 = calc_sma(ind["atr14"], 50)
    safe = np.where(np.isnan(atr_sma50) | (atr_sma50 < 1e-12), 1e-12, atr_sma50)
    ind["vol_regime"] = ind["atr14"] / safe
    red = (close < open_arr).astype(float)
    green = (close > open_arr).astype(float)
    ind["red_count5"] = pd.Series(red).rolling(5).sum().values
    ind["green_count5"] = pd.Series(green).rolling(5).sum().values
    for tn in [5, 10, 15, 20]:
        ind[f"rh_{tn}"] = pd.Series(high).rolling(tn + 1, min_periods=1).max().values
        ind[f"rl_{tn}"] = pd.Series(low).rolling(tn + 1, min_periods=1).min().values
    vs20 = ind["vol_sma20"]
    vs20_safe = np.where(np.isnan(vs20) | (vs20 <= 0), np.inf, vs20)
    ind["vf_pass"] = ~(~np.isnan(vs20) & (vs20 > 0) & (volume < 0.5 * vs20_safe))
    ind["_volume"] = volume
    sma20 = calc_sma(close, 20)
    std20 = calc_std(close, 20)
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20
    sma20_safe = np.where(np.abs(sma20) > 1e-12, sma20, 1e-12)
    ind["bb_width"] = (bb_upper - bb_lower) / sma20_safe
    ind["bw_p30"] = pd.Series(ind["bb_width"]).rolling(252, min_periods=50).quantile(0.30).values
    lag = np.full(n, np.nan)
    lag[10:] = sma20[:-10]
    lag_safe = np.where(np.abs(lag) > 1e-12, lag, 1e-12)
    ind["sma20_slope10"] = (sma20 - lag) / lag_safe
    return ind


def rolling_hl(high, low, tn):
    """rh/rl для произвольного N (тренд-RM с trail_n вне кэша, SW:1682-1683)."""
    rh = pd.Series(high).rolling(tn + 1, min_periods=1).max().values
    rl = pd.Series(low).rolling(tn + 1, min_periods=1).min().values
    return rh, rl
