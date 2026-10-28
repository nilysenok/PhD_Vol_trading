"""Подход B (V4:645-755; S5/S6 — V41:604-829): σ_pred вместо ATR поверх сигналов топ-10 A,
параметры исполнения — топ-1 A, выбор одной конфигурации B по Шарпу ансамбля на валидации.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from fastbt import grids
from fastbt.batch import (L_A1, L_A2, L_ADX, L_BE, L_CD, L_GAM, L_KIND, L_MH, L_SIG,
                          T_BASE, T_BTH, T_CD, T_GAM, T_ISL, T_KIND, T_SIG, T_TM, _lvl_one,
                          _trend_one)
from fastbt.data import year_bounds
from fastbt.execlayer import exec_layer, is_trivial, vote
from fastbt.stats import pos_from_trades, sharpe_window

HZ = ["h1", "h5", "h22"]


def b_matrix(cat):
    grid = grids.b_grid(cat)
    base = grids.B_BASE_HOLD[cat]
    if cat == "Trend":
        P = np.zeros((len(grid), 12))
        for q, bp in enumerate(grid):
            P[q, [T_KIND, T_ISL, T_TM, T_CD, T_BASE]] = [2, bp["k_sl"], bp["k_trail"], 5, base]
            P[q, T_BTH] = -1.0 if bp["k_be"] is None else bp["k_be"]
            P[q, T_GAM] = -1.0 if bp["gamma_hold"] is None else bp["gamma_hold"]
            P[q, T_SIG] = HZ.index(bp["horizon"])
        return grid, P
    P = np.zeros((len(grid), 11))
    for q, bp in enumerate(grid):
        P[q, [L_KIND, L_A1, L_A2, L_MH, L_CD]] = [2, bp["k_sl"], bp["k_sl"] * bp["ratio"], base, 5]
        P[q, L_ADX] = 1 if cat == "Range" else 0
        P[q, L_BE] = -1.0 if bp["k_be"] is None else bp["k_be"]
        P[q, L_GAM] = -1.0 if bp["gamma_hold"] is None else bp["gamma_hold"]
        P[q, L_SIG] = HZ.index(bp["horizon"])
    return grid, P


@njit(cache=True)
def _ens_member(trend, i_sp, r, sigs, exl, exs, nxt, close, high, low, adx, atr, atr0, sig3,
                med3, sdl, sds, tdl, tds, rh, rl, strength, ep, start, end, te, tx, td):
    n = len(close)
    if trend:
        nt = _trend_one(i_sp, r, sigs, exl, exs, nxt, close, high, low, atr, atr0, sig3, med3,
                        rh, rl, start, end, te, tx, td)
    else:
        nt = _lvl_one(i_sp, r, sigs, exl, exs, nxt, close, high, low, adx, atr0, sig3, med3,
                      sdl, sds, tdl, tds, start, end, te, tx, td)
    raw = pos_from_trades(te, tx, td, nt, n)
    if is_trivial(int(ep[0]), int(ep[1]), int(ep[2]), ep[3]):
        return raw
    return exec_layer(raw, strength[i_sp], int(ep[0]), int(ep[1]), int(ep[2]), ep[3], end)


@njit(cache=True, parallel=True)
def b_eval(trend, members, P, sigs, exl, exs, nxt, close, high, low, adx, atr, atr0, sig3,
           med3, sdl, sds, tdl, tds, rh, rl, strength, ep, lr, start, vs, ve, comm, out, frac):
    n = len(close)
    k = len(members)
    for q in prange(len(P)):
        te = np.empty(n, np.int64)
        tx = np.empty(n, np.int64)
        td = np.empty(n, np.int8)
        stack = np.zeros((k, n))
        for j in range(k):
            dup = -1
            for jj in range(j):
                if members[jj] == members[j]:
                    dup = jj
                    break
            if dup >= 0:
                stack[j, :] = stack[dup, :]
                continue
            stack[j, :] = _ens_member(trend, members[j], P[q], sigs, exl, exs, nxt, close, high,
                                      low, adx, atr, atr0, sig3, med3, sdl, sds, tdl, tds, rh,
                                      rl, strength, ep, start, ve, te, tx, td)
        ens = np.zeros(n)
        vote(stack, k, vs, ve, ens, frac)
        out[q] = sharpe_window(ens, lr, vs, ve, comm)


def _args(u):
    return (u.sigs, u.exl, u.exs, u.nxt, u.close, u.high, u.low, u.adx, u.atr, u.atr0)


def run_b(u, cfg, a_res):
    """a_res: {Y: (top10 [(i, r, e)], ensemble)}. Возвращает {Y: (bp, ensemble)}."""
    grid, P = b_matrix(u.cat)
    sig3 = np.stack([u.td.sigma[h] for h in HZ])
    rh, rl = (u.rh[:1], u.rl[:1]) if u.trend else (u.rh, u.rl)   # rh_10 — первый в TREND_TN
    eg = grids.exec_grid()
    out = {}
    b = cfg.bound()
    for Y in cfg.years_bcd:
        if Y not in a_res:
            continue
        top, _ = a_res[Y]
        vs, ve = cfg.val_bounds(u.dates, Y)
        if ve <= vs:
            continue
        lim = year_bounds(u.dates, cfg.cal_end_year(Y))[0] if cfg.fix_b_median else u.n
        med3 = np.array([_median(s[:lim]) for s in sig3])
        e0 = eg[top[0][2]]
        ep = np.array([e0["consec_entry"], e0["min_hold"], e0["cooldown_bars"],
                       e0["min_strength"]], np.float64)
        members = np.array([i for i, _, _ in top], np.int64)
        sh = np.full(len(P), -999.0)
        b_eval(u.trend, members, P, *_args(u), sig3, med3, u.sdl, u.sds, u.tdl, u.tds, rh, rl,
               u.strength, ep, u.lr, u.start, vs, ve, cfg.comm, sh, grids.VOTE_FRAC)
        q = int(np.argmax(sh)) if sh.max() > -999.0 else 0
        ens = _test_ensemble(u, P[q:q + 1], members, sig3, med3, rh, rl, ep)
        out[Y] = (grid[q], ens, float(sh[q]))
    return out


def _test_ensemble(u, P1, members, sig3, med3, rh, rl, ep):
    n = u.n
    te, tx, td = np.empty(n, np.int64), np.empty(n, np.int64), np.empty(n, np.int8)
    stack = np.stack([_ens_member(u.trend, int(m), P1[0], *_args(u), sig3, med3, u.sdl, u.sds,
                                  u.tdl, u.tds, rh, rl, u.strength, ep, u.start, n, te, tx, td)
                      for m in members])
    ens = np.zeros(n)
    vote(stack, len(members), 0, n, ens, grids.VOTE_FRAC)
    return ens


def _median(s):
    pos = s[s > 1e-6]
    return float(np.nanmedian(pos)) if len(pos) > 0 else 0.02
