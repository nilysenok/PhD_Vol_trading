"""Метрики из сделок: статистики по годовым сегментам и сборка окна из сумм.

Сегменты: [WARMUP, 1.01.2016), [2016), …, [2026, n). Для сегмента храним
N, S=Σnet, Q=Σnet², E=баров в позиции, R=входов из нуля, и моноид просадки
(Mx=max C, Mn=min C, D=min(C−cummax C)), C — накопленная сумма от начала сегмента.
net[t] = pos[t]·lr[t] − |pos[t]−pos[t−1]|·c  (V4:359-372). Окно A начинается с WARMUP,
где позиция всегда 0, поэтому сумма сегментов = счёт по маске (prepend=0).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

F_S, F_Q, F_E, F_R, F_MX, F_MN, F_D = range(7)
NF = 7


@njit(cache=True)
def _bar(out, cc, ini, segb, seg_of, t, net):
    k = seg_of[t]
    if not ini[k]:
        ini[k] = True
        cc[k] = 0.0
        if t > segb[k]:
            out[k, F_MX] = 0.0
            out[k, F_MN] = 0.0
        else:
            out[k, F_MX] = -np.inf
            out[k, F_MN] = np.inf
        out[k, F_D] = 0.0
    cc[k] += net
    c = cc[k]
    if c > out[k, F_MX]:
        out[k, F_MX] = c
    if c < out[k, F_MN]:
        out[k, F_MN] = c
    dd = c - out[k, F_MX]
    if dd < out[k, F_D]:
        out[k, F_D] = dd
    out[k, F_S] += net
    out[k, F_Q] += net * net


@njit(cache=True)
def seg_stats_trades(te, tx, tdir, nt, lr, segb, seg_of, end, comm, out):
    """out[NSEG, NF] (обнуляется здесь). Сделки упорядочены, позиция d на [e, x)."""
    ns = out.shape[0]
    out[:, :] = 0.0
    cc = np.zeros(ns)
    ini = np.zeros(ns, np.bool_)
    for i in range(nt):
        e = te[i]
        x = tx[i]
        d = float(tdir[i])
        pprev = 0.0
        if i > 0 and tx[i - 1] == e:
            pprev = float(tdir[i - 1])
        k0 = seg_of[e]
        if pprev == 0.0:
            out[k0, F_R] += 1.0
        _bar(out, cc, ini, segb, seg_of, e, d * lr[e] - abs(d - pprev) * comm)
        out[k0, F_E] += 1.0
        for t in range(e + 1, x):
            _bar(out, cc, ini, segb, seg_of, t, d * lr[t] - 0.0 * comm)
            out[seg_of[t], F_E] += 1.0
        if x < end and not (i + 1 < nt and te[i + 1] == x):
            _bar(out, cc, ini, segb, seg_of, x, 0.0 * lr[x] - abs(0.0 - d) * comm)


@njit(cache=True)
def seg_stats_pos(pos, lr, segb, seg_of, start, end, comm, out):
    """То же по ряду позиций (для рядов после слоя исполнения)."""
    ns = out.shape[0]
    out[:, :] = 0.0
    cc = np.zeros(ns)
    ini = np.zeros(ns, np.bool_)
    prev = 0.0
    for t in range(start, end):
        p = pos[t]
        if p != 0.0 or prev != 0.0:
            _bar(out, cc, ini, segb, seg_of, t, p * lr[t] - abs(p - prev) * comm)
        if p != 0.0:
            out[seg_of[t], F_E] += 1.0
            if prev == 0.0:
                out[seg_of[t], F_R] += 1.0
        prev = p


@njit(cache=True)
def pos_from_trades(te, tx, tdir, nt, n):
    pos = np.zeros(n)
    for i in range(nt):
        d = float(tdir[i])
        for t in range(te[i], tx[i]):
            pos[t] = d
    return pos


def window(stats, nbars, j, bpy=252.0):
    """Метрики окна из сегментов 0..j. stats[..., NSEG, NF]; nbars[NSEG].
    Возвращает (sharpe, maxdd, exposure_pct, trades_per_year)."""
    st = stats[..., :j + 1, :]
    N = float(nbars[:j + 1].sum())
    S = st[..., F_S].sum(-1)
    Q = st[..., F_Q].sum(-1)
    var = np.maximum(Q - S * S / N, 0.0) / (N - 1.0)
    sd = np.sqrt(var)
    with np.errstate(invalid="ignore", divide="ignore"):
        sh = np.where(sd < 1e-12, 0.0, (S / N) / sd * math.sqrt(252.0))
    acc_s = st[..., 0, F_S]
    acc_mx = st[..., 0, F_MX]
    acc_d = st[..., 0, F_D]
    for k in range(1, j + 1):
        mn = st[..., k, F_MN]
        acc_d = np.minimum(np.minimum(acc_d, st[..., k, F_D]), acc_s + mn - acc_mx)
        acc_mx = np.maximum(acc_mx, acc_s + st[..., k, F_MX])
        acc_s = acc_s + st[..., k, F_S]
    exp = st[..., F_E].sum(-1) / N * 100.0
    tpy = st[..., F_R].sum(-1) / max(N / bpy, 0.5)
    return sh, acc_d, exp, tpy


@njit(cache=True)
def sharpe_window(pos, lr, s, e, comm):
    """calc_sharpe_v4 на окне [s, e) (diff с prepend=0 от начала окна)."""
    n = e - s
    if n <= 0:
        return -999.0
    tot = 0.0
    prev = 0.0
    for t in range(s, e):
        p = pos[t]
        tot += p * lr[t] - abs(p - prev) * comm
        prev = p
    m = tot / n
    ss = 0.0
    prev = 0.0
    for t in range(s, e):
        p = pos[t]
        r = p * lr[t] - abs(p - prev) * comm - m
        ss += r * r
        prev = p
    sd = math.sqrt(ss / (n - 1)) if n > 1 else 0.0
    if sd < 1e-12:
        return 0.0
    return m / sd * math.sqrt(252.0)
