"""Трендовый движок (SW.bt_trend_v3 / bt_trend_vpred_v3), см. fastbt.engine."""
from __future__ import annotations

from numba import njit

from fastbt.engine import _dyn_hold


@njit(cache=True)
def trend_run(sig, exl, exs, nxt, close, high, low, atr, atr0, sigma, rh, rl,
              kind, isl, tmult, bth, tt, ps, pm, cd, base, gamma, med, start, end,
              te, tx, tdir):
    """Тренд: kind 0 — SW.bt_trend_v3 (без временного стопа), 2 — bt_trend_vpred_v3."""
    nt = 0
    t = start
    while t < end:
        e = nxt[t]
        if e >= end:
            break
        d = sig[e]
        ep = close[e]
        mht = 0
        if kind == 0:
            ea = atr0[e]
            sl = ep - isl * ea if d == 1 else ep + isl * ea
            ext = high[e] if d == 1 else low[e]
            step = ps
        else:
            es = sigma[e]
            if es != es:
                es = 0.02
            sl = ep * (1.0 - isl * es) if d == 1 else ep * (1.0 + isl * es)
            mht = _dyn_hold(base, es, med, gamma)
        be = False
        held = 0
        closed = False
        stop = False
        u = e + 1
        while u < end:
            held += 1
            if kind == 0:
                ca = atr[u]
                if ca != ca:
                    ca = ea
                if d == 1:
                    if tt < 2:
                        tsl = rh[u] - tmult * ca
                    elif tt == 3:           # ATR-trailing (Ж-127): максимум с входа, не окно N баров
                        ext = max(ext, high[u])
                        tsl = ext - tmult * ca
                    else:
                        if high[u] > ext:
                            ext = high[u]
                            step = min(step + ps, pm)
                        tsl = sl + step * (ext - sl)
                    if not be and bth < 1e10 and (close[u] - ep) > bth * ea:
                        be = True
                else:
                    if tt < 2:
                        tsl = rl[u] + tmult * ca
                    elif tt == 3:
                        ext = min(ext, low[u])
                        tsl = ext + tmult * ca
                    else:
                        if low[u] < ext:
                            ext = low[u]
                            step = min(step + ps, pm)
                        tsl = sl - step * (sl - ext)
                    if not be and bth < 1e10 and (ep - close[u]) > bth * ea:
                        be = True
            else:
                sv = sigma[u]
                if sv != sv:
                    sv = 0.02
                if d == 1:
                    tsl = rh[u] - tmult * sv * close[u]
                    if not be and bth > 0.0 and (close[u] - ep) > bth * es * ep:
                        be = True
                else:
                    tsl = rl[u] + tmult * sv * close[u]
                    if not be and bth > 0.0 and (ep - close[u]) > bth * es * ep:
                        be = True
            if d == 1:
                if be:
                    tsl = max(tsl, ep)
                sl = max(sl, tsl)
                if low[u] <= sl:
                    closed = True
                    stop = True
                elif exl[u]:
                    closed = True
            else:
                if be:
                    tsl = min(tsl, ep)
                sl = min(sl, tsl)
                if high[u] >= sl:
                    closed = True
                    stop = True
                elif exs[u]:
                    closed = True
            if not closed and kind == 2 and held >= mht:
                closed = True
            if closed:
                break
            u += 1
        te[nt] = e
        tx[nt] = u if closed else end
        tdir[nt] = d
        nt += 1
        if not closed:
            break
        t = u + cd if (stop and cd > 0) else u
    return nt
