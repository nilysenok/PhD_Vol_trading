"""Медленный эталон трендового выхода S3/S4 (Ж-127): чистый Python, без numba, написан отдельно от engine_trend.

trail_type: 0 fixed_atr и 1 chandelier — max(high, 11 баров с текущим) − k·ATR[t]; 2 parabolic_step;
3 atr_trail — max(high с бара входа по текущий включительно) − k·ATR[t] (пересчёт по срезу, без накопителя).
Храповик sl = max(sl, tsl); безубыток: close − ep > bth·ATR_вх → tsl не ниже ep. Выход по low ≤ sl (шорт — high ≥ sl)
на баре, исполнение по close бара. Возвращает [(вход, выход, знак)], как tests/audit_fastbt/_eng.run_trend.
"""
from __future__ import annotations

WIN = 11


def trend_medl(close, high, low, atr, sig, isl=2.0, tmult=2.0, bth=1e12, tt=0, ps=0.02, pm=0.2, cd=0):
    n = len(close)
    out, t = [], 0
    while t < n:
        e = next((i for i in range(t, n) if sig[i] != 0), n)
        if e >= n:
            break
        d, ep, ea = int(sig[e]), close[e], atr[e]
        sl = ep - d * isl * ea
        step, be, x = ps, False, None
        for u in range(e + 1, n):
            ca = atr[u]
            if tt < 2:
                lo_i = max(0, u - WIN + 1)
                tsl = max(high[lo_i:u + 1]) - tmult * ca if d == 1 else min(low[lo_i:u + 1]) + tmult * ca
            elif tt == 3:
                tsl = max(high[e:u + 1]) - tmult * ca if d == 1 else min(low[e:u + 1]) + tmult * ca
            else:
                ext = max(high[e:u + 1]) if d == 1 else min(low[e:u + 1])
                step = _step(high, low, e, u, d, ps, pm)
                tsl = sl + d * step * abs(ext - sl)
            if not be and bth < 1e10 and d * (close[u] - ep) > bth * ea:
                be = True
            if be:
                tsl = max(tsl, ep) if d == 1 else min(tsl, ep)
            sl = max(sl, tsl) if d == 1 else min(sl, tsl)
            if (low[u] <= sl) if d == 1 else (high[u] >= sl):
                x = u
                break
        out.append((e, n if x is None else x, d))
        if x is None:
            break
        t = x + cd if cd > 0 else x
    return out


def _step(high, low, e, u, d, ps, pm):
    """Шаг parabolic к бару u: растёт на ps при каждом новом экстремуме на барах e+1..u (потолок pm)."""
    step, ext = ps, (high[e] if d == 1 else low[e])
    for i in range(e + 1, u + 1):
        if (high[i] > ext) if d == 1 else (low[i] < ext):
            ext = high[i] if d == 1 else low[i]
            step = min(step + ps, pm)
    return step
