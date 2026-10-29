"""Быстрые (numba) версии фильтров v5 и рангов (лист 4 из П-0, #625). Эталон — исходные функции; сверка побайтно.

Объявлено заранее: допуск 0 (побитное равенство, nan на тех же местах). Суммы и средние начального окна ADX берутся numpy,
как в `indicators.calc_adx` (попарное суммирование numpy ≠ простой цикл numba), в numba — только рекуррентные циклы.
Причинность та же, что у эталонов: порог ATR — значения [0..d-1]; ранг — окно d-okno…d-1 (день d не входит).
Подключение в оценщик — не здесь.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from fastbt import rangi_v6
from fastbt.indicators import _adx_core, _dm_tr
from fastbt.signals.v5_filtr import MIN_ATR


@njit(cache=True)
def _adx_sglazhivanie(dx, adx, start, period):
    for i in range(start + 1, len(dx)):
        if not np.isnan(dx[i]) and not np.isnan(adx[i - 1]):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period


def calc_adx_nb(high, low, close, period=14):
    """= `indicators.calc_adx` побайтно; сглаживание в numba."""
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
        _adx_sglazhivanie(dx, adx, start, period)
    return adx


@njit(cache=True)
def _atr_expand(atr, pct, min_atr):
    n = len(atr)
    out = np.ones(n, np.bool_)
    if pct <= 0:
        return out
    out[:] = False
    srt = np.empty(n)
    m = 0
    for d in range(n):
        a = atr[d]
        if np.isnan(a):
            continue
        if m >= min_atr:
            pos = (m - 1) * pct / 100.0
            lo = int(pos)
            hi = min(lo + 1, m - 1)
            out[d] = a > srt[lo] + (srt[hi] - srt[lo]) * (pos - lo)
        k = np.searchsorted(srt[:m], a, side="right")     # insort = bisect_right
        srt[k + 1:m + 1] = srt[k:m].copy()
        srt[k] = a
        m += 1
    return out


def maska_atr_expand_nb(atr, pct):
    """= `signals.v5_filtr.maska_atr_expand`: ATR[d] > pct-й процентиль ATR[0..d-1]; pct <= 0 — выкл."""
    return _atr_expand(np.ascontiguousarray(atr, dtype=np.float64), float(pct), MIN_ATR)


@njit(cache=True)
def _rang_nb(m, pul, okno, min_nabl):
    nt, nd = m.shape
    out = np.full(m.shape, np.nan)
    ng = 1 if pul else nt
    buf = np.empty(nt * okno if pul else okno)
    for d in range(nd):
        lo = max(0, d - okno)
        for g in range(ng):
            r0, r1 = (0, nt) if pul else (g, g + 1)
            k = 0
            for i in range(r0, r1):
                for j in range(lo, d):
                    if np.isfinite(m[i, j]):
                        buf[k] = m[i, j]
                        k += 1
            if k < min_nabl:
                continue
            v = np.sort(buf[:k])
            for i in range(r0, r1):
                if np.isfinite(m[i, d]):
                    out[i, d] = np.searchsorted(v, m[i, d], side="left") / k
    return out


def rang_nb(m, tip: str, okno: int = rangi_v6.OKNO, min_nabl: int = rangi_v6.MIN_NABL) -> np.ndarray:
    """= `rangi_v6.rang` (tikernyj | obshchiy | srednee) побайтно."""
    if tip not in rangi_v6.RANG_TIPY:
        raise ValueError(f"rang_tip: допустимо {rangi_v6.RANG_TIPY}, получено {tip!r}")
    m = np.ascontiguousarray(m, dtype=np.float64)
    if tip == "srednee":
        return (_rang_nb(m, False, okno, min_nabl) + _rang_nb(m, True, okno, min_nabl)) / 2.0
    return _rang_nb(m, tip == "obshchiy", okno, min_nabl)
