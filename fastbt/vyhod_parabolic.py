"""Параболический трейлинг тренда (v4, `parabolic_step`) как отдельный выход.

Формула v4 (strategies_walkforward.py::bt_trend_v3, trail_type=2): стоп подтягивается
к экстремуму с входа на долю `step`: tsl = sl + step*(ext - sl) (лонг), sl = max(sl, tsl);
step стартует с ps и растёт на ps при каждом новом экстремуме, не выше pm.
Четыре уровня v4: ps in {0,01; 0,02} x pm in {0,10; 0,15}.

↺ от v4: в v4 экстремум и close[t] текущего бара входят в стоп того же бара
(стоп дня t знает high[t], а проверка low[t] идёт после — заглядывание внутри бара).
Здесь стоп бара u строится по состоянию на конец бара u-1; экстремум, шаг и
безубыток обновляются после проверки бара u. Режима «как в v4» нет.

Порядок в баре: стоп раньше сигнала выхода. Гэп через стоп: заполнение по открытию.
Деньги не считаются: возвращается индекс выхода и цена заполнения.
"""
from __future__ import annotations

import numpy as np
from numba import njit

NIVELI_V4 = ((0.01, 0.10), (0.01, 0.15), (0.02, 0.10), (0.02, 0.15))


@njit(cache=True)
def parabolic_exit(d, e, end, ep, ea, isl, ps, pm, bth,
                   opn, high, low, close, exit_sig):
    """Выход тренда: (u, fill, stop); u = -1 — позиция не закрыта до `end`.

    d = +1/-1, e — бар входа (цена ep, ATR входа ea), bth >= 1e10 — без безубытка.
    """
    sl = ep - isl * ea if d == 1 else ep + isl * ea
    ext = high[e] if d == 1 else low[e]
    step = ps
    be = False
    for u in range(e + 1, end):
        if d == 1:
            tsl = sl + step * (ext - sl)
            if be:
                tsl = max(tsl, ep)
            sl = max(sl, tsl)
            if low[u] <= sl:
                return u, min(sl, opn[u]), True
            if exit_sig[u]:
                return u, close[u], False
            if high[u] > ext:
                ext = high[u]
                step = min(step + ps, pm)
            if not be and bth < 1e10 and (close[u] - ep) > bth * ea:
                be = True
        else:
            tsl = sl - step * (sl - ext)
            if be:
                tsl = min(tsl, ep)
            sl = min(sl, tsl)
            if high[u] >= sl:
                return u, max(sl, opn[u]), True
            if exit_sig[u]:
                return u, close[u], False
            if low[u] < ext:
                ext = low[u]
                step = min(step + ps, pm)
            if not be and bth < 1e10 and (ep - close[u]) > bth * ea:
                be = True
    return -1, np.nan, False


def parabolic_exit_ref(d, e, end, ep, ea, isl, ps, pm, bth,
                       opn, high, low, close, exit_sig):
    """Медленный эталон: шорт зеркалится в лонг сменой знака цен (код не общий с numba)."""
    s = float(d)
    o, h, lo, c = s * np.asarray(opn), s * np.asarray(high), s * np.asarray(low), s * np.asarray(close)
    if d == -1:
        h, lo = -np.asarray(low), -np.asarray(high)
    p = s * ep
    sl = p - isl * ea
    ext, step, be = h[e], ps, False
    for u in range(e + 1, end):
        tsl = sl + step * (ext - sl)
        if be:
            tsl = max(tsl, p)
        sl = max(sl, tsl)
        if lo[u] <= sl:
            return u, s * min(sl, o[u]), True
        if exit_sig[u]:
            return u, close[u], False
        if h[u] > ext:
            ext, step = h[u], min(step + ps, pm)
        if not be and bth < 1e10 and c[u] - p > bth * ea:
            be = True
    return -1, float("nan"), False
