"""Общие части итераций 16–20 (T16–T20, Ж-065 п. 4; контракт #376 §2.1; толкование [Ж-065 п. 4], поправка 28.10 09:15).

z_d = r_d / σ̂_{d−1}, Z_d = R_d / sig_mkt_{d−1} (σ̂ — `sigma_real`, известна на 18:50 d, поэтому в знаменателе d−1).
Сила = |x| в σ (вся величина отклонения); вес w = min(L, (сила/порог)·0,20/(σ̂_d·√252)), фиксирован на сделку.
Вход — close Н6 дня e ≥ d+1 (Д2); событие принимается, только если e ≥ (последний день прошлой сделки) + 2.
"""
from __future__ import annotations

import numpy as np

TARGET = 0.20


def _ratio(num: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """num[d] / sigma[d−1]; sigma ≤ 0 или NaN → NaN; строка 0 — NaN."""
    num, s = np.asarray(num, float), np.asarray(sigma, float)
    if len(num) != len(s):
        raise ValueError("ряд и σ разной длины")
    out = np.full(len(num), np.nan)
    if len(num) > 1:
        prev = s[:-1]
        ok = np.isfinite(prev) & (prev > 0)
        out[1:][ok] = num[1:][ok] / prev[ok]
    return out


def z_ryad(close, sigma) -> np.ndarray:
    """z_d = ln(c_d / c_{d−1}) / σ̂_{d−1}; close ≤ 0 или NaN → NaN."""
    c = np.asarray(close, float)
    r = np.full(len(c), np.nan)
    if len(c) > 1:
        ok = np.isfinite(c[1:]) & np.isfinite(c[:-1]) & (c[1:] > 0) & (c[:-1] > 0)
        r[1:][ok] = np.log(c[1:][ok] / c[:-1][ok])
    return _ratio(r, sigma)


def Z_ryad(imoex_close, sig_mkt) -> np.ndarray:
    """Z_d = R_d / sig_mkt_{d−1}, R_d — доходность IMOEX по close 18:50 (тот же закон, что у z)."""
    return z_ryad(imoex_close, sig_mkt)


def sila(x):
    """Сила сигнала = |x| в σ."""
    return np.abs(np.asarray(x, float))


def ves(sila_x, porog: float, sigma_d, L: float = 1.0, target: float = TARGET):
    """w = min(L, (сила/порог)·target/(σ̂_d·√252)); σ̂ ≤ 0 или NaN, сила NaN → 0. Пропорционален силе до лимита L."""
    if not porog > 0:
        raise ValueError("порог должен быть > 0")
    s, sg = np.asarray(sila_x, float), np.asarray(sigma_d, float)
    good = np.isfinite(s) & np.isfinite(sg) & (sg > 0)
    raw = (np.where(good, s, 0.0) / porog) * target / (np.where(good, sg, 1.0) * np.sqrt(252.0))
    w = np.minimum(np.where(good, raw, 0.0), L)
    return float(w) if w.ndim == 0 else w


def postroit_sig(n: int, events, tau: int, sigma, porog: float, lag: int = 1, L: float = 1.0):
    """События одного тикера [(d, сторона ±1, сила)] → (sig, w, принятые [(d, e, сторона, вес)]).

    sig[e..e+τ] = сторона (обрезка по n), w[e] = вес по σ̂_d. e = d + lag, lag ≥ 1 (Д2). Принимается событие
    с e ≥ конец_прошлой + 2, конец = e_прошлой + τ; отброшенные сделку не продлевают. Вес 0 / NaN — событие пропущено."""
    if lag < 1:
        raise ValueError("lag ≥ 1: исполнение не раньше d+1 (Д2)")
    if tau < 0:
        raise ValueError("tau ≥ 0")
    sigma = np.asarray(sigma, float)
    sig, w, acc, end = np.zeros(n, int), np.zeros(n), [], -10 ** 9
    for d, side, sl in sorted(events, key=lambda ev: ev[0]):
        e = d + lag
        if side not in (1, -1) or not 0 <= d < n or e >= n or e < end + 2:
            continue
        wt = float(ves(sl, porog, sigma[d], L))
        if not wt > 0:
            continue
        sig[e:e + tau + 1], w[e] = side, wt
        acc.append((d, e, side, wt))
        end = e + tau
    return sig, w, acc
