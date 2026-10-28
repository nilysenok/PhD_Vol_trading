"""Семейство S1 MeanRev + S2 Bollinger (daily): побитовый перенос gen_s1/gen_s2_signals и
signal_strength_s1/s2 из c22a4b3 (SW:489-534, V4:128-146). Спецификация: п. 11, 12.

SMA/STD окна — fastbt.indicators (pandas rolling, как в старом коде: биты порогов те же),
по одному разу на окно. Фильтры входа (vol_regime, красные/зелёные, объём, vf_pass) не зависят
от параметров — считаются один раз. Конфигурации — numba-ядра, O(K*T).
"""
from __future__ import annotations

import sys

import numpy as np
from numba import njit

from fastbt.indicators import calc_sma, calc_std
from fastbt.signals.interface import WARMUP, Family, SignalSet, register

GRID_S1 = dict(ma_window=[15, 20, 25], z_entry=[1.5, 2.0, 2.5], z_exit=[0.3, 0.5, 0.7])
GRID_S2 = dict(bb_window=[15, 20, 25], bb_std=[1.5, 2.0, 2.5])


def _filters(ind):
    """Общие фильтры входа S1/S2: (long, short) bool[T]."""
    vr = ind["vol_regime"]
    f_vr = ~np.isnan(vr) & (vr < 1.2)
    rc, gc = ind["red_count5"], ind["green_count5"]
    vs5, vol = ind["vol_sma5"], ind["_volume"]
    vs5_safe = np.where(np.isnan(vs5) | (vs5 <= 0), np.inf, vs5)
    f_ve = ~(~np.isnan(vs5) & (vs5 > 0) & (vol >= vs5_safe))
    base = f_vr & f_ve & ind["vf_pass"]
    return base & ~np.isnan(rc) & (rc >= 3), base & ~np.isnan(gc) & (gc >= 3)


@njit(cache=True)
def _s1(close, sma, std, fl, fs, z_entry, z_exit, sig, ex, st):
    for t in range(len(close)):
        z = (close[t] - sma[t]) / std[t]
        if np.isnan(z):
            st[t] = np.nan
            continue
        if t >= WARMUP:
            if z < -z_entry and fl[t]:
                sig[t] = 1
            elif z > z_entry and fs[t]:
                sig[t] = -1
        ex[t] = abs(z) < z_exit
        st[t] = min(abs(z) / 3.0, 1.0)


@njit(cache=True)
def _s2(close, sma, std, fl, fs, k, sig, ex, st):
    for t in range(len(close)):
        if np.isnan(sma[t]):
            st[t] = np.nan
            continue
        lo = sma[t] - k * std[t]
        up = sma[t] + k * std[t]
        if t >= WARMUP:
            if close[t] <= lo and fl[t]:
                sig[t] = 1
            elif close[t] >= up and fs[t]:
                sig[t] = -1
        st[t] = min(max(max(lo - close[t], close[t] - up), 0.0) / max(std[t], 1e-12) / 0.5, 1.0)
        if t > 0:  # sma[t-1] = NaN: сравнения ложны, как в старом коде
            pc, pm = close[t - 1], sma[t - 1]
            ex[t] = (pc < pm and close[t] >= sma[t]) or (pc > pm and close[t] <= sma[t])


def _make(kind):
    wkey = "ma_window" if kind == 1 else "bb_window"

    def generate(td, ind, grid):
        K, T = len(grid), td.n
        sig = np.zeros((K, T), np.int8)
        ex = np.zeros((K, T), np.bool_)
        st = np.zeros((K, T))
        fl, fs = _filters(ind)
        cache = {}
        for i, p in enumerate(grid):
            w = p[wkey]
            if w not in cache:
                cache[w] = (calc_sma(td.close, w), calc_std(td.close, w))
            sma, std = cache[w]
            if kind == 1:
                _s1(td.close, sma, std, fl, fs, p["z_entry"], p["z_exit"], sig[i], ex[i], st[i])
            else:
                _s2(td.close, sma, std, fl, fs, p["bb_std"], sig[i], ex[i], st[i])
        return SignalSet(sig, ex, ex, st, list(grid), split=False)
    return generate


_LEG = {}


def _legacy(sid):
    def legacy(L, td, p):
        key = (td.ticker, id(td))
        if key not in _LEG:
            _LEG.clear()
            _LEG[key] = L.sw.compute_base(td.close, td.high, td.low, td.open, td.volume)
        ind = _LEG[key]
        w = p["ma_window"] if sid == "S1" else p["bb_window"]
        cache = L.sw.precompute_sma_cache(td.close, [w])
        gen = L.sw.gen_s1_signals if sid == "S1" else L.sw.gen_s2_signals
        sig, ex = gen(td.close, ind, cache, **p)
        st = L.v4.dispatch_strength(sid, td.close, ind, cache, {}, {}, {}, None, p)
        return sig, ex, None, st
    return legacy


_me = sys.modules[__name__]
S1 = register(Family("S1", "Contrarian", GRID_S1, _make(1), _legacy("S1"), module=_me))
S2 = register(Family("S2", "Contrarian", GRID_S2, _make(2), _legacy("S2"), module=_me))
