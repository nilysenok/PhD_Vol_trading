"""Слой исполнения V4 (V4:211-275) и голосование (V4:627-631) — numba, побитово.

Порядок: фильтр силы → |pos|>0,5 → consec_entry → min_hold → cooldown → знак сырой позиции.
Голосование: mag = mean|pos_k| > 0,5 (≡ 2·#ненулевых > K), dir = sign(Σ pos_k).
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def is_trivial(consec, min_hold, cool, min_str):
    return consec <= 1 and min_hold <= 0 and cool <= 0 and min_str <= 0.0


@njit(cache=True)
def exec_layer(raw, strength, consec, min_hold, cool, min_str, end):
    """Возвращает позиции {−1,0,+1} на [0, end) (дальше нули)."""
    n = len(raw)
    b = np.zeros(n)
    for i in range(end):
        p = abs(raw[i])
        if min_str > 0.0 and strength[i] < min_str:
            p = 0.0
        b[i] = 1.0 if p > 0.5 else 0.0
    if consec > 1:
        cnt = 0
        inp = False
        for i in range(end):
            if b[i] > 0:
                if inp:
                    b[i] = 1.0
                else:
                    cnt += 1
                    if cnt >= consec:
                        b[i] = 1.0
                        inp = True
                    else:
                        b[i] = 0.0
            else:
                cnt = 0
                inp = False
    if min_hold > 0:
        rem = 0
        for i in range(end):
            if b[i] > 0 and rem == 0:
                rem = min_hold
            if rem > 0:
                b[i] = 1.0
                rem -= 1
            else:
                b[i] = 0.0
    if cool > 0:
        res = np.zeros(n)
        cl = 0
        for i in range(end):
            if cl > 0:
                cl -= 1
                continue
            if b[i] > 0:
                res[i] = 1.0
            elif i > 0 and b[i] == 0 and res[i - 1] > 0:
                cl = cool
        b = res
    out = np.zeros(n)
    for i in range(end):
        r = raw[i]
        s = 1.0 if r > 0 else (-1.0 if r < 0 else 0.0)
        out[i] = b[i] * s
    return out


@njit(cache=True)
def vote(stack, k, s, e, out, frac=0.5):
    """stack[k, n]; ансамбль на [s, e) пишется в out (вне окна не трогается).
    Голос есть, если ненулевых больше frac·k (код: frac = 0,5)."""
    for t in range(s, e):
        cnt = 0
        tot = 0.0
        for j in range(k):
            v = stack[j, t]
            if v != 0.0:
                cnt += 1
            tot += v
        if cnt > frac * k:
            out[t] = 1.0 if tot > 0 else (-1.0 if tot < 0 else 0.0)
        else:
            out[t] = 0.0
