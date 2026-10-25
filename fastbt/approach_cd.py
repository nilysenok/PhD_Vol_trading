"""Подходы C и D: булевы фильтры над позицией ансамбля A (приём 9).

Маска закрывает позицию A побарно (final = a_pos · mask), поэтому пересчёт идёт только
по барам, где A в позиции (и бару выхода); остальные бары дают ровно 0. Отбор —
первый строго лучший Шарп на валидации (V4:762-979, V41:835-967); у C — ограничения
экспозиции/частоты сделок в два прохода.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from fastbt import grids
from fastbt.data import year_bounds
from fastbt.masks import c_masks, d_gates


@njit(cache=True, parallel=True)
def masked_eval(apos, masks, lr, vs, ve, comm, sh, ex, tpy):
    n_win = ve - vs
    act = np.empty(n_win, np.int64)
    na = 0
    for t in range(vs, ve):
        if apos[t] != 0.0 or (t > vs and apos[t - 1] != 0.0):
            act[na] = t
            na += 1
    years = max(n_win / 252.0, 0.5)
    for m in prange(masks.shape[0]):
        mk = masks[m]
        S = 0.0
        Q = 0.0
        E = 0
        R = 0
        for j in range(na):
            t = act[j]
            q = apos[t] if mk[t] else 0.0
            p = 0.0
            if t > vs and mk[t - 1]:
                p = apos[t - 1]
            net = q * lr[t] - abs(q - p) * comm
            S += net
            Q += net * net
            if q != 0.0:
                E += 1
                if p == 0.0:
                    R += 1
        var = max(Q - S * S / n_win, 0.0) / (n_win - 1)
        sd = math.sqrt(var)
        sh[m] = 0.0 if sd < 1e-12 else (S / n_win) / sd * math.sqrt(252.0)
        ex[m] = E / n_win * 100.0
        tpy[m] = R / years


def _eval(apos, rows, lr, vs, ve, comm):
    M = rows.shape[0]
    sh, ex, tp = np.zeros(M), np.zeros(M), np.zeros(M)
    masked_eval(apos, rows, lr, vs, ve, comm, sh, ex, tp)
    return sh, ex, tp


def _first_best(sh, ok=None):
    best, bi = -999.0, -1
    for i in range(len(sh)):
        if sh[i] > best and (ok is None or ok[i]):
            best, bi = sh[i], i
    return bi


def run_c(u, cfg, a_res):
    """{Y: (params|'A', позиции)}; отсутствие ключа Y = строки C пропущены (18.3)."""
    v41 = u.sid in ("S1", "S2", "S5", "S6")
    rows, idx, params = c_masks(u.td, u.cat, v41)
    limits = grids.C_LIMITS_V41 if v41 else grids.C_LIMITS_V4
    b = cfg.bound()
    out = {}
    for Y in cfg.years_bcd:
        if Y not in a_res:
            continue
        ens = a_res[Y][1]
        apos = ens
        if u.sid in ("S1", "S2") and not cfg.fix_c_s12_val:
            ys, ye = year_bounds(u.dates, Y)[0], year_bounds(u.dates, Y + 1)[0]
            apos = np.zeros(u.n)
            apos[ys:ye] = ens[ys:ye]               # V41:1446-1478: A только тестового года
        vs, ve = year_bounds(u.dates, cfg.val_start)[b], year_bounds(u.dates, Y)[b]
        sh, ex, tp = _eval(apos, rows, u.lr, vs, ve, cfg.comm)
        sh, ex, tp = sh[idx], ex[idx], tp[idx]
        bi = -1
        for mexp, mtpy in limits:
            bi = _first_best(sh, (ex >= mexp) & (tp >= mtpy))
            if bi >= 0:
                break
        if bi < 0:
            if v41 or cfg.fix_c_fill:
                out[Y] = ("A", apos.copy(), None)
            continue
        out[Y] = (params[bi], apos * rows[idx[bi]], float(sh[bi]))
    return out


def run_d(u, cfg, a_res):
    rows, idx, params = d_gates(u.td, cfg.fix_d_units)
    b = cfg.bound()
    out = {}
    for Y in cfg.years_bcd:
        if Y not in a_res:
            continue
        apos = a_res[Y][1]
        vs, ve = year_bounds(u.dates, cfg.val_start)[b], year_bounds(u.dates, Y)[b]
        sh, _, _ = _eval(apos, rows, u.lr, vs, ve, cfg.comm)
        sh = sh[idx]
        bi = _first_best(sh)
        if bi < 0:
            continue
        out[Y] = (params[bi], apos * rows[idx[bi]].astype(np.float64), float(sh[bi]))
    return out
