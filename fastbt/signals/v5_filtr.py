"""Фильтры Р по типу (сетка v5, Ж-127): обёртка над fastbt/filtry_v5.py.

API: primenit(df, tip, params, sig=None) -> bool[n] (True = вход на баре d разрешён).
  df   — DataFrame с колонками high, low, close, volume и датами (DatetimeIndex или колонка "date");
  tip  — "kontr" | "trend" (zh126_obshchee.yaml: tip_strategii); params — adx_th, atr_lo, vol_th, rsi_th, mom_n, dow,
         squeeze_pct (отсутствует/0 = выкл.).
  sig  — сигнал int8[n]; нужен, если включён rsi_th или mom_n (направление): маска запрещает бар, где источник
         обнулил бы sig[d] (daily_upper_bound.py:302-342). Без sig и при включённом rsi/mom — ValueError.
Причинность: маска[d] зависит от данных <= d. ATR-процентиль — расширяющееся окно ДО d-1 (порог из значений
ATR[0..d-1], > 50 значений, иначе запрет); filtry_v5.maska_atr_pctl (скользящее, включая d) не используется.
Сжатие — filtry_v5.maska_szhatie (252, окно включает d: сигнал на закрытии d, исполнение >= d+1). Отступление от
источника: при <= 50 значений ATR источник фильтр не применял, здесь запрет (иначе зависит от длины ряда).
"""
from __future__ import annotations

from bisect import insort

import numpy as np
import pandas as pd

from fastbt.filtry_v5 import maska_rsi, maska_szhatie, rsi_wilder, shirina_bb
from fastbt.indicators import calc_adx, calc_atr, calc_sma

MIN_ATR = 51


def maska_atr_expand(atr, pct):
    """ATR[d] > pct-й процентиль (линейная интерполяция, как np.percentile) значений ATR[0..d-1]; pct <= 0 — выкл."""
    atr = np.asarray(atr, dtype=float)
    out = np.ones(len(atr), bool)
    if pct <= 0:
        return out
    out[:] = False
    srt: list = []
    for d in range(len(atr)):
        m = len(srt)
        if m >= MIN_ATR and not np.isnan(atr[d]):
            pos = (m - 1) * pct / 100.0
            lo = int(pos)
            hi = min(lo + 1, m - 1)
            thr = srt[lo] + (srt[hi] - srt[lo]) * (pos - lo)
            out[d] = atr[d] > thr
        if not np.isnan(atr[d]):
            insort(srt, float(atr[d]))
    return out


def _dni_nedeli(df):
    idx = df.index if hasattr(df.index, "dayofweek") else None
    if idx is None:
        idx = pd.DatetimeIndex(df["date"])
    return np.asarray(idx.dayofweek)


def primenit(df, tip, params, sig=None):
    if tip not in ("kontr", "trend"):
        raise ValueError(f"tip: {tip!r}")
    trend = tip == "trend"
    close = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    vol = df["volume"].to_numpy(float)
    n = len(close)
    mask = np.ones(n, bool)
    at = params.get("adx_th", 0)
    if at > 0:
        adx = calc_adx(high, low, close, 14)
        with np.errstate(invalid="ignore"):
            mask &= ~np.isnan(adx) & ((adx > at) if trend else (adx < at))
    al = params.get("atr_lo", 0)
    if al > 0:
        mask &= maska_atr_expand(calc_atr(high, low, close, 14), al)
    vt = params.get("vol_th", 0)
    if vt > 0:
        vs = calc_sma(vol, 20)
        mask &= vol / np.where(np.isnan(vs) | (vs < 1), 1.0, vs) > vt
    sp = params.get("squeeze_pct", 0)
    if sp > 0:
        mask &= maska_szhatie(shirina_bb(close), 252, sp / 100.0, 50)
    dw = params.get("dow", 0)
    if dw > 0:
        d = _dni_nedeli(df)
        mask &= {1: d != 0, 2: d != 4, 3: (d != 0) & (d != 4)}[dw]
    rt, mn = params.get("rsi_th", 0), params.get("mom_n", 0)
    if rt > 0 or mn > 0:
        if sig is None:
            raise ValueError("rsi_th/mom_n зависят от направления: нужен sig")
        mask &= _napravlenie(sig, close, trend, rt, mn)
    return mask


def _napravlenie(sig, close, trend, rt, mn):
    """False там, где sig[d] != 0 и направление отвергнуто RSI (источник 320-328) или mom (329-334)."""
    sig = np.asarray(sig)
    ok_l = np.ones(len(sig), bool)
    ok_s = np.ones(len(sig), bool)
    if rt > 0:
        ok_l, ok_s = maska_rsi(rsi_wilder(close, 14), rt, po_trendu=trend)
    if mn > 0:
        mom = np.zeros(len(close))
        mom[mn:] = close[mn:] - close[:-mn]
        ok_l = ok_l & ((mom > 0) if trend else (mom < 0))
        ok_s = ok_s & ((mom < 0) if trend else (mom > 0))
    return ~(((sig == 1) & ~ok_l) | ((sig == -1) & ~ok_s))
