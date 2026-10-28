"""Семейство S5 (Pivot Points) и S6 (VWAP), daily — V41 @ c22a4b3 (spec п. 11, S5R:245-297, S6R:201-276).

Индикаторы — numba, O(T) на конфигурацию окна/пивота; pandas-арифметика старого кода
воспроизведена побитово (Kahan-суммы rolling sum/std pandas, FMA как в скомпилированном pandas).
Данные: дневные бары (volume = сумма 10-минуток за день), внутридневных данных нет;
VWAP_w = rolling-сумма tp*vol / rolling-сумма vol по w ДНЕВНЫМ барам (min_periods=w).
"""
from __future__ import annotations

import hashlib
import inspect
import sys

import numpy as np
import pandas as pd

from fastbt.indicators import calc_sma
from fastbt.signals.interface import Family, SignalSet, register
from fastbt.signals import s56_kern as _kern
from fastbt.signals.s56_kern import _entry, _pivots, _s5_exit, _vwap


def _filters(ind, fs):
    """Наборы фильтров S5 (fs 2,3,4) и S6 (fs 1..4), S5R:245-297 / S6R:217-276; плюс vf_pass."""
    adx, sl, vr = ind["adx14"], ind["sma20_slope10"], ind["vol_regime"]
    with np.errstate(invalid="ignore"):
        a30, a25 = ~np.isnan(adx) & (adx < 30), ~np.isnan(adx) & (adx < 25)
        f_sl2 = ~np.isnan(sl) & (np.abs(sl) < 0.02)
        f_sl1 = ~np.isnan(sl) & (np.abs(sl) < 0.01)
        f_vr = lambda c: ~np.isnan(vr) & (vr < c)
        f_bw = ~np.isnan(ind["bw_p30"]) & (ind["bb_width"] < ind["bw_p30"])
        f = {1: ~np.isnan(adx) & (adx < 35), 2: a30 & f_sl2, 3: a30 & f_sl2 & f_vr(0.95),
             4: a25 & f_bw & f_sl1 & f_vr(0.9)}[fs]
    return f & ind["vf_pass"]


def _gid(dates, period):
    ix = pd.DatetimeIndex(dates)
    if period == "weekly":
        c = ix.isocalendar()
        return (c["year"].values.astype(np.int64) * 100 + c["week"].values.astype(np.int64))
    return ix.year.values.astype(np.int64) * 100 + ix.month.values.astype(np.int64)


def generate5(td, ind, grid):
    n, K = td.n, len(grid)
    sig = np.zeros((K, n), np.int8); ex = np.zeros((K, n), bool)
    st = np.zeros((K, n)); extra = {k: np.zeros((K, n)) for k in ("P", "S1", "S2", "R1", "R2")}
    piv, flt = {}, {}
    for i, p in enumerate(grid):
        key = (p["pivot_period"], p["pivot_type"])
        if key not in piv:
            piv[key] = _pivots(_gid(td.dates, key[0]), td.high, td.low, td.close, key[1] == "fibonacci")
        if p["filter_set"] not in flt:
            flt[p["filter_set"]] = _filters(ind, p["filter_set"])
        P, S1, S2, R1, R2 = piv[key]
        with np.errstate(invalid="ignore"):
            _entry(sig[i], td.close < S1, td.close > R1, flt[p["filter_set"]])
            span = np.abs(R1 - S1)
            dist = np.minimum(np.abs(td.close - S1), np.abs(td.close - R1))
            st[i] = np.minimum(np.where(np.isnan(dist), 0.0, dist)
                               / np.where(np.isnan(span) | (span < 1e-12), 1e-12, span), 1.0)
        ex[i] = _s5_exit(td.close, P)
        for k, a in zip(("P", "S1", "S2", "R1", "R2"), piv[key]):
            extra[k][i] = a
    return SignalSet(sig, ex, ex.copy(), st, list(grid), False, extra)


def generate6(td, ind, grid):
    n, K = td.n, len(grid)
    sig = np.zeros((K, n), np.int8); ex = np.zeros((K, n), bool); st = np.zeros((K, n))
    extra = {"vwap": np.zeros((K, n)), "dev": np.zeros((K, n))}
    vc, flt = {}, {}
    ma50, ma200 = calc_sma(td.close, 50), calc_sma(td.close, 200)   # dir_bias: build_daily_trend
    bull = ~np.isnan(ma50) & ~np.isnan(ma200) & (ma50 > ma200)
    bear = ~np.isnan(ma50) & ~np.isnan(ma200) & ~(ma50 > ma200)
    for i, p in enumerate(grid):
        w, m, fs = p["vwap_window"], p["dev_mult"], p["filter_set"]
        if w not in vc:
            vc[w] = _vwap(td.close, td.high, td.low, td.volume, w)
        if fs not in flt:
            flt[fs] = _filters(ind, fs)
        vwap, dev = vc[w]
        with np.errstate(invalid="ignore"):
            lc = ~np.isnan(vwap) & (td.close < vwap - m * dev)
            sc = ~np.isnan(vwap) & (td.close > vwap + m * dev)
            if p["dir_bias"] == "ON":
                lc, sc = lc & bull, sc & bear
            _entry(sig[i], lc, sc, flt[fs])
            ex[i] = ~np.isnan(vwap) & (np.abs(td.close - vwap) < 0.5 * dev)
            st[i] = np.minimum(np.abs(td.close - np.where(np.isnan(vwap), td.close, vwap))
                               / (2.0 * np.maximum(dev, 1e-12)), 1.0)
        extra["vwap"][i], extra["dev"][i] = vwap, dev
    return SignalSet(sig, ex, ex.copy(), st, list(grid), False, extra)


_CACHE = {}


def _legacy_ctx(L, td):
    """Старый конвейер V41 process_ticker_v41 (daily): ind, pivot_cache, vwap_cache, dir_bias."""
    if td.ticker not in _CACHE:
        from fastbt.data import daily_df
        df = daily_df()
        tdf = df[df["ticker"] == td.ticker].sort_values("date").reset_index(drop=True)
        ind = L.sw.compute_base(td.close, td.high, td.low, td.open, td.volume, False)
        pc = L.v41.prepare_pivot_cache(df, td.ticker, None, False, tdf)
        vc, _, dba = L.v41.prepare_vwap_cache_ext(td.close, td.high, td.low, td.volume,
                                                  False, tdf, df, td.ticker)
        _CACHE[td.ticker] = (ind, pc, vc, dba)
    return _CACHE[td.ticker]


def legacy5(L, td, p):
    ind, pc, _, _ = _legacy_ctx(L, td)
    P, S1, S2, R1, R2 = pc[(p["pivot_period"], p["pivot_type"])]
    sig, ex = L.s5r.gen_s5_signals_v2(td.close, ind, P, S1, R1, p["filter_set"])
    span = np.abs(R1 - S1); dist = np.minimum(np.abs(td.close - S1), np.abs(td.close - R1))
    st = np.minimum(np.where(np.isnan(dist), 0.0, dist)
                    / np.where(np.isnan(span) | (span < 1e-12), 1e-12, span), 1.0)
    return sig, ex, None, st


def legacy6(L, td, p):
    ind, _, vc, dba = _legacy_ctx(L, td)
    vwap, dev = vc[p["vwap_window"]]
    sig, ex = L.s6r.gen_s6_signals_v2(td.close, ind, vwap, dev, p["dev_mult"], p["filter_set"],
                                      dba if p["dir_bias"] == "ON" else None)
    st = np.minimum(np.abs(td.close - np.where(np.isnan(vwap), td.close, vwap))
                    / (2.0 * np.maximum(dev, 1e-12)), 1.0)
    return sig, ex, None, st


G5 = dict(pivot_period=["weekly", "monthly"], pivot_type=["classic", "fibonacci"], filter_set=[2, 3, 4])
G6 = dict(vwap_window=[5, 10, 20, 60], dev_mult=[0.5, 1.0, 1.5, 2.0],
          filter_set=[1, 2, 3, 4], dir_bias=["OFF", "ON"])
_M = sys.modules[__name__]
_KV = hashlib.sha1(inspect.getsource(_kern).encode()).hexdigest()[:12]  # хэш ядер -> ключ кэша
S5 = register(Family("S5", "Range", G5, generate5, legacy5, code_version="1-" + _KV, module=_M))
S6 = register(Family("S6", "Range", G6, generate6, legacy6, code_version="1-" + _KV, module=_M))
