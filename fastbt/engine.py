"""Движки сделок (numba). Результат конфигурации — список сделок (вход, выход, знак).

Семантика — ровно SW.bt_*_v3 / *_vpred_v3, S5R.bt_range_pivot_v3, S6R.bt_range_vwap_v3
(c22a4b3; спецификация пп. 2–7, 14): вход/выход по close, стоп/тейк — триггеры по
low/high, стоп раньше тейка, cooldown только после стопа, повторный вход на баре выхода.
Плоские участки пропускаются прыжком по nxt[t] (ближайший бар с сигналом ≥ t).

kind: 0 — ATR (уровни ±k·ATR[вход]), 1 — расстояния pivot/VWAP, 2 — σ_pred (B).
Выход x — бар выхода (позиция d на [e, x)); x == end — сделка открыта на конце.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def next_signal(sig):
    n = len(sig)
    nxt = np.empty(n, np.int64)
    j = n
    for t in range(n - 1, -1, -1):
        if sig[t] != 0:
            j = t
        nxt[t] = j
    return nxt


@njit(cache=True)
def _dyn_hold(base, sv, med, gamma):
    if gamma > 0.0 and sv > 1e-8 and med > 1e-8:
        mh = base * (med / sv) ** gamma
        return int(min(max(mh, 3.0), base * 3.0))
    return int(base)


@njit(cache=True)
def lvl_run(sig, exl, exs, nxt, close, high, low, adx, atr0, sigma, sdl, sds, tdl, tds,
            kind, a1, a2, mh, be, cd, tdec, adx_on, gamma, med, start, end, te, tx, tdir):
    """Контртренд / диапазон (ATR, уровни, σ). Возвращает число сделок."""
    nt = 0
    t = start
    while t < end:
        e = nxt[t]
        if e >= end:
            break
        d = sig[e]
        c = close[e]
        otp = 0.0
        bthr = 0.0
        mht = int(mh)
        if kind == 0:
            a = atr0[e]
            if d == 1:
                sl = c - a1 * a
                tp = c + a2 * a
            else:
                sl = c + a1 * a
                tp = c - a2 * a
            otp = a2 * a
            bthr = be * a
        elif kind == 1:
            if d == 1:
                sd = sdl[e]
                tdd = tdl[e]
            else:
                sd = sds[e]
                tdd = tds[e]
            if sd != sd:
                sd = 0.001 * c
            if tdd != tdd:
                tdd = 0.001 * c
            if d == 1:
                sl = c - sd
                tp = c + tdd
            else:
                sl = c + sd
                tp = c - tdd
            otp = tdd
            bthr = be * tdd
        else:
            sv = sigma[e]
            if sv != sv:
                sv = 0.02
            if d == 1:
                sl = c * (1.0 - a1 * sv)
                tp = c * (1.0 + a2 * sv)
            else:
                sl = c * (1.0 + a1 * sv)
                tp = c * (1.0 - a2 * sv)
            mht = _dyn_hold(mh, sv, med, gamma)
        ep = c
        be_act = False
        held = 0
        closed = False
        stop = False
        u = e + 1
        while u < end:
            held += 1
            if be > 0.0 and not be_act:
                thr = bthr
                if kind == 2:
                    sv2 = sigma[u]
                    if sv2 != sv2:
                        sv2 = 0.02
                    thr = be * sv2 * ep
                if d == 1 and (close[u] - ep) > thr:
                    be_act = True
                    sl = ep
                elif d == -1 and (ep - close[u]) > thr:
                    be_act = True
                    sl = ep
            if tdec > 0 and held > mht // 2:
                tp = ep + 0.5 * otp if d == 1 else ep - 0.5 * otp
            if d == 1:
                if low[u] <= sl:
                    closed = True
                    stop = True
                elif high[u] >= tp:
                    closed = True
            else:
                if high[u] >= sl:
                    closed = True
                    stop = True
                elif low[u] <= tp:
                    closed = True
            if not closed and adx_on > 0:
                av = adx[u]
                if av == av and av > 30.0:
                    closed = True
            if not closed:
                if (d == 1 and exl[u]) or (d == -1 and exs[u]):
                    closed = True
            if not closed and held >= mht:
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
