"""Семь типов скользящих для осей S1/S2 сетки v5 (Ж-127): SMA, EMA, WMA, HMA, DEMA, TEMA, KAMA.

Формулы — как в защищённой версии (коммит 6a7a4636, moex_volatility/scripts/daily_upper_bound.py,
строки 46–91): EMA α = 2/(n+1), старт x[0], без NaN-хвоста; WMA веса 1..n, NaN первые n-1;
HMA (в источнике «hull») = WMA(2·WMA(n//2) − WMA(n), max(int(√n),1)); DEMA = 2·E1 − E2;
KAMA fast=2, slow=30, старт x[n-1], ER=0 при пути < 1e-12; SMA — pandas rolling, min_periods=n.
TEMA — расширение по Ж-125 п. 1, в защищённом коде нет: 3·E1 − 3·E2 + E3 на том же EMA (старт x[0]).
Значение на дне d зависит только от x[≤ d]. Движку и сеткам не подключено.
"""

from math import sqrt

import numpy as np
import pandas as pd
from numba import njit

TIPY = ("sma", "ema", "wma", "hma", "dema", "tema", "kama")
_SINONIMY = {"hull": "hma"}


def sma(x, n):
    return pd.Series(np.asarray(x, float)).rolling(n, min_periods=n).mean().values


def ema(x, n):
    x = np.asarray(x, float)
    a = 2.0 / (n + 1)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def wma(x, n):
    x = np.asarray(x, float)
    w = np.arange(1, n + 1, dtype=np.float64)
    out = np.full_like(x, np.nan)
    for i in range(n - 1, len(x)):
        out[i] = np.dot(x[i - n + 1:i + 1], w) / w.sum()
    return out


def kama(x, n, fast=2, slow=30):
    x = np.asarray(x, float)
    fc, sc = 2.0 / (fast + 1), 2.0 / (slow + 1)
    out = np.full_like(x, np.nan)
    if len(x) < n:
        return out
    out[n - 1] = x[n - 1]
    for i in range(n, len(x)):
        d = abs(x[i] - x[i - n])
        v = np.sum(np.abs(np.diff(x[i - n:i + 1])))
        er = 0 if v < 1e-12 else d / v
        out[i] = out[i - 1] + (er * (fc - sc) + sc) ** 2 * (x[i] - out[i - 1])
    return out


@njit(cache=True)
def sma_nb(x, n):
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        s = 0.0
        for j in range(i - n + 1, i + 1):
            s += x[j]
        out[i] = s / n
    return out


@njit(cache=True)
def ema_nb(x, n):
    a = 2.0 / (n + 1)
    out = np.empty(len(x))
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


@njit(cache=True)
def wma_nb(x, n):
    out = np.full(len(x), np.nan)
    zn = n * (n + 1) / 2.0
    for i in range(n - 1, len(x)):
        s = 0.0
        for k in range(n):
            s += x[i - n + 1 + k] * (k + 1)
        out[i] = s / zn
    return out


@njit(cache=True)
def kama_nb(x, n, fast, slow):
    fc, sc = 2.0 / (fast + 1), 2.0 / (slow + 1)
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    out[n - 1] = x[n - 1]
    for i in range(n, len(x)):
        d = abs(x[i] - x[i - n])
        v = 0.0
        for j in range(i - n + 1, i + 1):
            v += abs(x[j] - x[j - 1])
        er = 0.0 if v < 1e-12 else d / v
        out[i] = out[i - 1] + (er * (fc - sc) + sc) ** 2 * (x[i] - out[i - 1])
    return out


def _sostav(tip, x, n, f_sma, f_ema, f_wma):
    if tip == "sma":
        return f_sma(x, n)
    if tip == "ema":
        return f_ema(x, n)
    if tip == "wma":
        return f_wma(x, n)
    if tip == "dema":
        e1 = f_ema(x, n)
        return 2 * e1 - f_ema(e1, n)
    if tip == "tema":
        e1 = f_ema(x, n)
        e2 = f_ema(e1, n)
        return 3 * e1 - 3 * e2 + f_ema(e2, n)
    if tip == "hma":
        diff = 2 * f_wma(x, max(n // 2, 1)) - f_wma(x, n)
        return f_wma(diff, max(int(sqrt(n)), 1))
    raise ValueError(f"неизвестный тип MA: {tip!r}")


def ma(x, tip, okno, numba=False, **dop):
    """MA типа tip с окном okno; dop — fast/slow для KAMA (по умолчанию 2/30)."""
    tip = _SINONIMY.get(tip, tip)
    x = np.ascontiguousarray(x, dtype=np.float64)
    if tip not in TIPY:
        raise ValueError(f"неизвестный тип MA: {tip!r}")
    if tip == "kama":
        fast, slow = dop.get("fast", 2), dop.get("slow", 30)
        return kama_nb(x, okno, fast, slow) if numba else kama(x, okno, fast, slow)
    if dop:
        raise ValueError(f"параметры {sorted(dop)} не для {tip}")
    if numba:
        return _sostav(tip, x, okno, sma_nb, ema_nb, wma_nb)
    return _sostav(tip, x, okno, sma, ema, wma)
