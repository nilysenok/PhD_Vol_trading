"""Фильтры входа по типу (Ж-127, сетка v5): RSI, ATR-процентиль, наклон MA, сжатие полос.

Каждый фильтр — маска входа mask[d] из данных <= d (включая закрытие дня d); решение исполняется не раньше d+1
(Д2, не здесь). Формулы — из защищённого кода 6a7a4636: strategies_walkforward.py (calc_rsi 226-246, calc_atr 248-256,
compute_base 379-391) и daily_upper_bound.py (apply_filters_generic 302-328, calc_ema 46-51); сжатие — _range_filter 779-781.
NaN в любом входе маски => запрет (как в источнике). К движку и сеткам не подключено.

Объявлено до кода: порог ATR-процентиля и квантиль ширины — скользящее окно, оканчивающееся в d ВКЛЮЧИТЕЛЬНО
(сигнал на закрытии d, исполнение >= d+1). В источнике (daily_upper_bound.py:311-314) процентиль ATR — по ВСЕМУ ряду
(заглядывание вперёд, Ж-127 / инвентаризация #568) — не перенесён; тест-мутация «по всему ряду» ловится.
Кельтнера в источнике нет — расширение (ширина 2·mult·ATR / EMA).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fastbt.indicators import calc_atr, calc_sma, calc_std
from fastbt.ma_tipy import ema as _ema


def calc_ema(arr, w):
    """EMA с alpha = 2/(w+1), out[0] = arr[0] (daily_upper_bound.py:46-51)."""
    return _ema(arr, w)   # старт с первого конечного значения (rel-ряд с NaN прогрева)


MA_FUNCS = {"sma": calc_sma, "ema": calc_ema}   # тип MA — функция (arr, w); 7 типов ma_tipy — подставит следующий лист


def rsi_wilder(close, period=14):
    """RSI Уайлдера (strategies_walkforward.py:226-246): первое значение на индексе period, дальше сглаживание 1/period."""
    close = np.asarray(close, dtype=float)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    n = len(close)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi
    ag = np.mean(gain[1:period + 1])
    al = np.mean(loss[1:period + 1])
    rsi[period] = 100.0 if al < 1e-12 else 100.0 - 100.0 / (1.0 + ag / al)
    for i in range(period + 1, n):
        ag = (ag * (period - 1) + gain[i]) / period
        al = (al * (period - 1) + loss[i]) / period
        rsi[i] = 100.0 if al < 1e-12 else 100.0 - 100.0 / (1.0 + ag / al)
    return rsi


def maska_rsi(rsi, porog, po_trendu=False):
    """(long, short). Контртренд: long при rsi < porog, short при rsi > 100-porog; по тренду: long при rsi > 50,
    short при rsi < 50 (daily_upper_bound.py:321-328). porog <= 0 — фильтр выключен (всё разрешено). NaN — запрет."""
    rsi = np.asarray(rsi, dtype=float)
    ok = ~np.isnan(rsi)
    if porog <= 0:
        return np.ones(len(rsi), bool), np.ones(len(rsi), bool)
    with np.errstate(invalid="ignore"):
        if po_trendu:
            return ok & (rsi > 50), ok & (rsi < 50)
        return ok & (rsi < porog), ok & (rsi > 100 - porog)


def porog_kvantil(x, m, q, min_n):
    """Скользящий квантиль q (0..1, линейная интерполяция = np.percentile) по окну m, оканчивающемуся в d (d включён)."""
    return pd.Series(np.asarray(x, dtype=float)).rolling(m, min_periods=min_n).quantile(q).values


def atr_veroyatnost(atr, m, pct, min_n=51):
    """Порог ATR[d]: pct-й процентиль ATR в окне m дней, оканчивающемся в d."""
    return porog_kvantil(atr, m, pct / 100.0, min_n)


def maska_atr_pctl(atr, m, pct, min_n=51):
    """Вход разрешён, если ATR[d] > pct-й процентиль ATR за окно m (строго >, как в источнике 314). pct <= 0 — выкл."""
    atr = np.asarray(atr, dtype=float)
    if pct <= 0:
        return np.ones(len(atr), bool)
    thr = atr_veroyatnost(atr, m, pct, min_n)
    with np.errstate(invalid="ignore"):
        return ~np.isnan(atr) & ~np.isnan(thr) & (atr > thr)


def naklon_ma(close, n=20, k=10, tip="sma"):
    """Наклон MA: (MA[d] - MA[d-k]) / MA[d-k] (|MA[d-k]| < 1e-12 -> 1e-12), compute_base 379-391 при n=20, k=10, sma."""
    ma = MA_FUNCS[tip](np.asarray(close, dtype=float), n) if isinstance(tip, str) else tip(np.asarray(close, float), n)
    lag = np.full(len(ma), np.nan)
    lag[k:] = ma[:-k]
    lag_safe = np.where(np.abs(lag) > 1e-12, lag, 1e-12)
    return (ma - lag) / lag_safe


def maska_naklon(slope, porog, rezhim="abs"):
    """(long, short). "abs" — боковик: |наклон| < porog (источник 782-784, porog=0,01) для обеих сторон;
    "po_trendu": long при наклоне > porog, short при наклоне < -porog. NaN — запрет."""
    slope = np.asarray(slope, dtype=float)
    ok = ~np.isnan(slope)
    with np.errstate(invalid="ignore"):
        if rezhim == "abs":
            m = ok & (np.abs(slope) < porog)
            return m, m.copy()
        if rezhim == "po_trendu":
            return ok & (slope > porog), ok & (slope < -porog)
    raise ValueError(f"rezhim: {rezhim!r}")


def shirina_bb(close, n=20, k=2.0):
    """Ширина Боллинджера: 2k·std(ddof=1) / SMA (compute_base 379-384, n=20, k=2)."""
    close = np.asarray(close, dtype=float)
    sma = calc_sma(close, n)
    sma_safe = np.where(np.abs(sma) > 1e-12, sma, 1e-12)
    return ((sma + k * calc_std(close, n)) - (sma - k * calc_std(close, n))) / sma_safe


def shirina_keltner(high, low, close, n=20, mult=2.0):
    """Ширина Кельтнера (расширение, в источнике нет): 2·mult·ATR(n) / EMA(n)."""
    ema = calc_ema(close, n)
    ema_safe = np.where(np.abs(ema) > 1e-12, ema, 1e-12)
    return 2.0 * mult * calc_atr(np.asarray(high, float), np.asarray(low, float), np.asarray(close, float), n) / ema_safe


def maska_szhatie(shirina, m=252, q=0.30, min_n=50):
    """Сжатие: ширина[d] < q-квантиль ширины за окно m, оканчивающееся в d (источник 779-781: 252, 0,30, min_periods=50).
    Для обеих сторон. NaN — запрет."""
    w = np.asarray(shirina, dtype=float)
    thr = porog_kvantil(w, m, q, min_n)
    with np.errstate(invalid="ignore"):
        return ~np.isnan(thr) & (w < thr)
