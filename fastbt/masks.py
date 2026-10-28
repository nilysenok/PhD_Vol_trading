"""Маски C (SW._build_c_mask_v3, SW:1931-1994) и затворы D (SW._compute_d_scale_v3,
V4:923-966) для тикера — в порядке перебора старого кода, с дедупликацией по хэшу (приём 10).

Маски не зависят ни от стратегии, ни от года: строятся один раз на тикер.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from numba import njit

from fastbt import grids


@njit(cache=True)
def hysteresis(pr, above, entry, exit_, n):
    mask = np.zeros(n, dtype=np.bool_)
    active = False
    for i in range(n):
        v = pr[i]
        if v != v:
            mask[i] = False
            continue
        if above == 1:
            if not active and v > entry:
                active = True
            elif active and v < exit_:
                active = False
        else:
            if not active and v < (1.0 - entry):
                active = True
            elif active and v > (1.0 - exit_):
                active = False
        mask[i] = active
    return mask


@njit(cache=True)
def hyst_gate(r, th_in, th_out):
    """Затвор с гистерезисом: открыт; закрыть при r > th_out; открыть при r < th_in.
    r = NaN (нет истории) — затвор открыт."""
    n = len(r)
    g = np.ones(n, dtype=np.bool_)
    st = True
    for i in range(n):
        v = r[i]
        if v != v:
            st = True
        elif st and v > th_out:
            st = False
        elif (not st) and v < th_in:
            st = True
        g[i] = st
    return g


class Dedup:
    """Список масок в порядке перебора → уникальные строки + индекс."""

    def __init__(self):
        self.rows, self.idx, self._h = [], [], {}

    def add(self, m):
        h = hashlib.sha1(np.packbits(m).tobytes()).digest()
        if h not in self._h:
            self._h[h] = len(self.rows)
            self.rows.append(m)
        self.idx.append(self._h[h])

    def done(self):
        return np.ascontiguousarray(np.stack(self.rows)), np.array(self.idx, np.int64)


def _c_mask(cat, pr, sig, thr, direction, s22, s1, term, hyst, n):
    if cat == "Trend":
        m = (hysteresis(pr, 1, thr, thr - hyst / 100.0, n) if hyst > 0
             else ~np.isnan(pr) & (pr > thr))
    elif cat == "Contrarian":
        m = (hysteresis(pr, 0, thr, thr - hyst / 100.0, n) if hyst > 0
             else ~np.isnan(pr) & (pr < (1 - thr)))
    else:
        lo, hi = thr
        m = ~np.isnan(pr) & (pr >= lo) & (pr <= hi)
    if direction != "OFF":
        ds = np.full(n, np.nan)
        ds[5:] = sig[5:] - sig[:-5]
        m = m & ~np.isnan(ds) & ((ds > 0) if direction == "rise" else (ds < 0))
    if term:
        ratio = s22 / np.maximum(s1, 1e-8)
        ok = ~np.isnan(ratio)
        if cat == "Trend":
            m = m & ok & (ratio > 1.0)
        elif cat == "Contrarian":
            m = m & ok & (ratio < 1.0)
    return m


_CACHE = {}


def _sigma_digest(td) -> str:
    """Маски C и затворы D строятся из td.sigma — σ входит в ключ (аудит FA2b-Н2)."""
    h = hashlib.sha1()
    for k in sorted(td.sigma):
        h.update(k.encode() + np.ascontiguousarray(td.sigma[k], dtype=np.float64).tobytes())
    return h.hexdigest()


def _cached(name, td, *args):
    k = (name, td.ticker, td.digest(), _sigma_digest(td)) + args
    if k not in _CACHE:
        _CACHE[k] = globals()[f"_{name}"](td, *args)
    return _CACHE[k]


def c_masks(td, cat, v41):
    return _cached("c_masks", td, cat, v41)


def d_gates(td, annualize):
    return _cached("d_gates", td, annualize)


def _c_masks(td, cat, v41):
    """(уникальные маски [M, T], индекс конфигурации → строка, список параметров).
    Ранг σ — по окну [t−L, t−1] без текущего дня (s9_stats.past_rank; аудит FA, Н1, Ъ ~23:20:
    ручки нет). Ранг c22a4b3 (pandas rolling.rank с текущим днём) не воспроизводится."""
    from fastbt.s9_stats import past_rank
    sg, n = td.sigma, td.n
    pr = {(h, lb): past_rank(np.ascontiguousarray(sg[h], dtype=np.float64), lb, max(50, lb // 5))
          for h in grids.C_HORIZONS for lb in grids.C_LOOKBACKS}
    if cat == "Range":
        outer = grids.C_RANGE_BANDS_V41 if v41 else grids.C_RANGE_BANDS
    else:
        outer = grids.C_THRESHOLDS_V41 if v41 else grids.C_THRESHOLDS
    dd, params = Dedup(), []
    for thr in outer:
        for lb in grids.C_LOOKBACKS:
            for h in grids.C_HORIZONS:
                for dr in (grids.C_DIRECTION_TREND if cat == "Trend" else grids.C_DIRECTION):
                    for tf in grids.C_TERM_FILTER:
                        if tf and h == "h22":
                            continue
                        for hy in (grids.C_HYSTERESIS_TREND if cat == "Trend"
                                   else grids.C_HYSTERESIS):
                            dd.add(_c_mask(cat, pr[(h, lb)], sg[h], thr, dr, sg["h22"], sg["h1"],
                                           tf, hy, n))
                            params.append(dict(thr=thr, lookback=lb, horizon=h, direction=dr,
                                               term_filter=tf, hysteresis=hy))
    rows, idx = dd.done()
    return rows, idx, params


def _d_sigma(raw, smooth, vf, vc):
    s = raw.copy()
    if smooth is not None:
        s = pd.Series(s).ewm(span=smooth, min_periods=1).mean().values
    if vf is not None:
        s = np.maximum(s, vf)
    if vc is not None:
        s = np.minimum(s, vc)
    return np.where(np.isnan(s) | (s < 1e-6), 1e-6, s)


def _d_gates_rel(td):
    """Относительный затвор (доводка D): r = σ_pred / медиана σ_pred тикера за предыдущие
    D_REL_WINDOW баров (сдвиг на 1: только прошлое, min 63). Нет r — затвор открыт.
    rel: открыт при r < θ; hyst: закрыть при r > θ_out, открыть при r < θ_in."""
    dd, params = Dedup(), []
    for h in grids.D_HORIZONS:
        s = pd.Series(td.sigma[h])
        med = s.shift(1).rolling(grids.D_REL_WINDOW, min_periods=63).median()
        r = (s / med).values.astype(np.float64)
        if grids.D_GATE_MODE == "rel":
            for th in grids.D_REL_THR:
                dd.add(np.isnan(r) | (r < th))
                params.append(dict(mode="rel", horizon=h, theta=th))
        else:
            for ti in grids.D_HYST_IN:
                for gap in grids.D_HYST_GAP:
                    to = round(ti + gap, 4)
                    dd.add(hyst_gate(r, ti, to))
                    params.append(dict(mode="hyst", horizon=h, theta_in=ti, theta_out=to))
    rows, idx = dd.done()
    return rows, idx, params


def _d_gates(td, annualize):
    """Затворы D в порядке V4:924-966: стандартный вариант, затем обратная волатильность."""
    if grids.D_GATE_MODE in ("rel", "hyst"):
        return _d_gates_rel(td)
    sg = {h: (td.sigma[h] * np.sqrt(252.0) if annualize else td.sigma[h]) for h in grids.D_HORIZONS}
    dd, params = Dedup(), []
    for tv in (grids.D_TARGET_VOLS if grids.D_STANDARD else []):
        for ml in grids.D_MAX_LEVS:
            for g in grids.D_GAMMA:
                for vf in grids.D_VOL_FLOORS:
                    for vc in grids.D_VOL_CAPS:
                        for sm in grids.D_SMOOTH:
                            for h in grids.D_HORIZONS:
                                s = _d_sigma(sg[h], sm, vf, vc)
                                scale = np.minimum((tv / s) ** g, ml)
                                for th in grids.D_GATE_THRESHOLDS:
                                    dd.add(scale >= th)
                                    params.append(dict(target_vol=tv, max_lev=ml, gamma=g,
                                                       vol_floor=vf, vol_cap=vc, smooth_span=sm,
                                                       horizon=h, inverse_vol=False,
                                                       gate_threshold=th))
    for lb in (grids.D_INV_LOOKBACKS if grids.D_INVERSE else []):
        for ml in grids.D_MAX_LEVS:
            for sm in grids.D_SMOOTH:
                for h in grids.D_HORIZONS:
                    s = _d_sigma(sg[h], sm, None, None)
                    inv = 1.0 / s
                    norm = pd.Series(inv).rolling(lb, min_periods=20).mean().values
                    norm = np.where(np.isnan(norm) | (norm < 1e-6), 1e-6, norm)
                    scale = np.minimum(inv / norm, ml)
                    for th in grids.D_GATE_THRESHOLDS:
                        dd.add(scale >= th)
                        params.append(dict(max_lev=ml, inv_lookback=lb, smooth_span=sm,
                                           horizon=h, inverse_vol=True, gate_threshold=th))
    rows, idx = dd.done()
    return rows, idx, params
