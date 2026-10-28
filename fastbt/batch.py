"""Пакетный прогон конфигураций (numba prange): сделки → статистики по сегментам.

Каждая конфигурация = (индекс сигнала, строка RM). Сделки не хранятся — сразу
сводятся в статистики сегментов (приёмы 7–8). Для финалистов — trades_one().
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from fastbt.engine import lvl_run
from fastbt.engine_trend import trend_run
from fastbt.stats import NF, seg_stats_trades

# столбцы RM-матрицы уровневого движка
L_KIND, L_A1, L_A2, L_MH, L_BE, L_CD, L_TD, L_ADX, L_DIST, L_GAM, L_SIG = range(11)
# столбцы RM-матрицы трендового движка
T_KIND, T_ISL, T_TM, T_BTH, T_TT, T_PS, T_PM, T_CD, T_RH, T_BASE, T_GAM, T_SIG = range(12)


@njit(cache=True)
def _lvl_one(i_sp, r, sigs, exl, exs, nxt, close, high, low, adx, atr0, sig3, med3,
             sdl, sds, tdl, tds, start, end, te, tx, td):
    kind = int(r[L_KIND])
    di = int(r[L_DIST])
    hs = int(r[L_SIG])
    return lvl_run(sigs[i_sp], exl[i_sp], exs[i_sp], nxt[i_sp], close, high, low, adx, atr0,
                   sig3[hs], sdl[i_sp, di], sds[i_sp, di], tdl[i_sp, di], tds[i_sp, di],
                   kind, r[L_A1], r[L_A2], r[L_MH], r[L_BE], int(r[L_CD]), int(r[L_TD]),
                   int(r[L_ADX]), r[L_GAM], med3[hs], start, end, te, tx, td)


@njit(cache=True)
def _trend_one(i_sp, r, sigs, exl, exs, nxt, close, high, low, atr, atr0, sig3, med3,
               rh, rl, start, end, te, tx, td):
    hs = int(r[T_SIG])
    k = int(r[T_RH])
    return trend_run(sigs[i_sp], exl[i_sp], exs[i_sp], nxt[i_sp], close, high, low, atr, atr0,
                     sig3[hs], rh[k], rl[k], int(r[T_KIND]), r[T_ISL], r[T_TM], r[T_BTH],
                     int(r[T_TT]), r[T_PS], r[T_PM], int(r[T_CD]), r[T_BASE], r[T_GAM],
                     med3[hs], start, end, te, tx, td)


@njit(cache=True, parallel=True)
def grid_lvl(cfg_sp, cfg_rm, rmP, sigs, exl, exs, nxt, close, high, low, adx, atr0, sig3,
             med3, sdl, sds, tdl, tds, lr, segb, seg_of, comm, start, end, out):
    n = len(close)
    for q in prange(len(cfg_sp)):
        te = np.empty(n, np.int64)
        tx = np.empty(n, np.int64)
        td = np.empty(n, np.int8)
        nt = _lvl_one(cfg_sp[q], rmP[cfg_rm[q]], sigs, exl, exs, nxt, close, high, low, adx,
                      atr0, sig3, med3, sdl, sds, tdl, tds, start, end, te, tx, td)
        seg_stats_trades(te, tx, td, nt, lr, segb, seg_of, end, comm, out[q])


@njit(cache=True, parallel=True)
def grid_trend(cfg_sp, cfg_rm, rmP, sigs, exl, exs, nxt, close, high, low, atr, atr0, sig3,
               med3, rh, rl, lr, segb, seg_of, comm, start, end, out):
    n = len(close)
    for q in prange(len(cfg_sp)):
        te = np.empty(n, np.int64)
        tx = np.empty(n, np.int64)
        td = np.empty(n, np.int8)
        nt = _trend_one(cfg_sp[q], rmP[cfg_rm[q]], sigs, exl, exs, nxt, close, high, low, atr,
                        atr0, sig3, med3, rh, rl, start, end, te, tx, td)
        seg_stats_trades(te, tx, td, nt, lr, segb, seg_of, end, comm, out[q])


def run_grid(u, cfg_sp, cfg_rm, rmP, trend, comm, sig3=None, med3=None):
    """Статистики [K, NSEG, NF] для пар (сигнал, строка RM)."""
    cfg_sp = np.asarray(cfg_sp, np.int64)
    cfg_rm = np.asarray(cfg_rm, np.int64)
    out = np.zeros((len(cfg_sp), len(u.nbars), NF))
    sig3 = u.sig3 if sig3 is None else sig3
    med3 = u.med3 if med3 is None else med3
    if trend:
        grid_trend(cfg_sp, cfg_rm, rmP, u.sigs, u.exl, u.exs, u.nxt, u.close, u.high, u.low,
                   u.atr, u.atr0, sig3, med3, u.rh, u.rl, u.lr, u.segb, u.seg_of, comm,
                   u.start, u.n, out)
    else:
        grid_lvl(cfg_sp, cfg_rm, rmP, u.sigs, u.exl, u.exs, u.nxt, u.close, u.high, u.low,
                 u.adx, u.atr0, sig3, med3, u.sdl, u.sds, u.tdl, u.tds, u.lr, u.segb,
                 u.seg_of, comm, u.start, u.n, out)
    return out


def trades_one(u, i_sp, r, trend, end=None, sig3=None, med3=None):
    """Список сделок одной конфигурации на [WARMUP, end)."""
    end = u.n if end is None else end
    te = np.empty(u.n, np.int64)
    tx = np.empty(u.n, np.int64)
    td = np.empty(u.n, np.int8)
    sig3 = u.sig3 if sig3 is None else sig3
    med3 = u.med3 if med3 is None else med3
    if trend:
        nt = _trend_one(i_sp, r, u.sigs, u.exl, u.exs, u.nxt, u.close, u.high, u.low, u.atr,
                        u.atr0, sig3, med3, u.rh, u.rl, u.start, end, te, tx, td)
    else:
        nt = _lvl_one(i_sp, r, u.sigs, u.exl, u.exs, u.nxt, u.close, u.high, u.low, u.adx,
                      u.atr0, sig3, med3, u.sdl, u.sds, u.tdl, u.tds, u.start, end, te, tx, td)
    return te[:nt], tx[:nt], td[:nt]
