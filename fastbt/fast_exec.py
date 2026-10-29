"""Быстрое ядро exec_e (Ж-083): тот же автомат, что `exec_e_run.Run`, одним numba-проходом по дням и барам.

Правила и арифметика — построчно как `exec_e_run` / `exec_e_rules` (порядок операций тот же, без fastmath),
поэтому результат побайтно равен медленному пути; медленный путь — эталон (proverka.md п. 3).
Лимит входа и выхода — только при проходе на тик (l ≤ L − тик, h ≥ L + тик; касание не исполняет, mutations № 2),
цена — L или open при гэпе (№ 4); в баре стоп проверяется раньше тейка (№ 3). Вход в день d+1, не раньше (Д2).
Состояние автомата — массивы fs (float) и st (int); сделки — массивы ti/tf, обёртка — fastbt/fast_etap1.run.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numba import njit

from fastbt import fast_exec_parab as pb
from fastbt import fast_exec_v6 as v6

EPS = 1e-6                                              # = exec_e_rules.EPS
REF, SG, EXT, SH, DV, OL, OSGO, XX, EPX, SGT, SMAX, W, RR, OUP, ODN = range(15)   # fs (SGT — единица тейка, SMAX — храповик, W — остаток, RR — R)
TR, SD, I0, EJ, BE, BLK, OK, ODUE, OSD, ODS, ORT, XO, XR, SR, NO, NF, XN, XF, NT = range(19)   # st
CUR, ENTD, ENTB, PART, OAGE, LXD, LXB, LXR, DNRE, PQ = range(19, 29)   # st: текущая сделка, бар входа, ½ взята, возраст заявки, последний выход, день повтора, «стоп в плюс» включён
NAD, AD0, AD1 = 29, 30, 31                              # st: число добавок pir, индексы их ног в ti (−1 — нет)
XT, PEND = 32, 33                                       # st: выходов не по close (t_vyhod open2); выход по сигналу отложен на open x+1
NST = 35                                                # + PBN = 34 (флаг бара входа parab, fast_exec_parab)
P_PARAB, F_PS, F_PM = pb.P_PARAB, pb.F_PS, pb.F_PM
F_PIR, P_PIR = v6.F_PIR, v6.P_PIR
BEM, VLEV, TA = 15, 16, 17                              # fs: порог безубытка в σ, уровень v6-стопа (храповик), a тейка (ось a_rang); fast_exec_v6
F_B0, F_B0C, F_SP, F_SQ, F_STEP, F_AA0, F_AA1 = v6.F_B0, v6.F_B0C, v6.F_SP, v6.F_SQ, v6.F_STEP, v6.F_AA0, v6.F_AA1
P_B0, P_B0C, P_SP, P_STEP, P_ARANG = v6.P_B0, v6.P_B0C, v6.P_SP, v6.P_STEP, v6.P_ARANG
DS, ED, EB, XD, XB, RS, SDC, R0, PAR = range(9)         # ti (PAR — индекс родителя частичного выхода, −1)
F_K, F_A, F_B, F_COMM, F_TICK, F_R, F_H, F_KTR, F_NAL, F_CH, F_KE, F_ATRM, F_CF, F_CR = range(14)       # pf
(P_HK, P_F, P_HA, P_HB, P_HTAU, P_TAU, P_EZH, P_EXS, P_SIDE, P_HTICK, P_TOUCH, P_HR, P_HH, P_HKTR, P_BE,
 P_ZAK, P_REENTRY, P_VLIM, P_US, P_UT, P_CH, P_SZ, P_ATRM, P_T, P_RED, P_CHAST) = range(26)   # pi (P_US/P_UT — код ED, P_CH — chand,
# P_SZ — stop_zayavka, P_ATRM — atr_m задан, P_T — срок заявки, P_RED — reentry_den, P_CHAST — частичный выход)
P_TV = 32                                               # pi: t_vyhod = open2 (после v6: P_B0…P_PIR = 26…31)
REASONS = (None, "stop", "trail", "bu", "take", "vremya", "signal", "lim", "kanal", "part", "pir")
R_STOP, R_TRAIL, R_BU, R_TAKE, R_VREMYA, R_SIGNAL, R_LIM, R_KANAL, R_PART, R_PIR = range(1, 11)
YADRO_S34_GOTOV = True      # оси stop_zayavka+urovni, atr_m, T, reentry_den, chast_1r/chast в ядре (включены в pokryto, #658)
YADRO_PARAB_GOTOV = True    # Ф-parab: ось parab (П3b, параболический трейлинг) в ядре; снят — такие точки идут эталоном
YADRO_OSI3_GOTOV = True     # Ф-сл3d: оси #653 (b0, b0_c, sp, step_s, a_rang + вход rang) и #660 (pir) в ядре; снят — эти точки идут эталоном
F_CODES = {"propusk": 0, "zakrytie_d1": 1, "povtor": 2}
ED_CODES = {"sigma": 0, "atr": 1, "pct": 2}            # Л5: единица уровней (exec_e.unit_series)
ATR_N, CH_OKNO = 14, 11                                 # = exec_e_rules.atr_series n; окно экстремума Run._chand


@njit(cache=True)
def _floor_t(x, tick):
    return math.floor(x / tick + EPS) * tick if tick != 0.0 else x


@njit(cache=True)
def _ceil_t(x, tick):
    return math.ceil(x / tick - EPS) * tick if tick != 0.0 else x


@njit(cache=True)
def _limit_level(sd, close, sg, pf, pi):
    L = close * (1 - sd * pf[F_K] * sg)
    if not pi[P_HTICK]:
        return L
    return _floor_t(L, pf[F_TICK]) if sd > 0 else _ceil_t(L, pf[F_TICK])


@njit(cache=True)
def _limit_hit(sd, h, l, L, pf, pi):
    t = pf[F_TICK]
    if sd > 0:
        return l < L if not pi[P_HTICK] else l <= L - t + t * EPS
    return h > L if not pi[P_HTICK] else h >= L + t - t * EPS


@njit(cache=True)
def _take_hit(sd, h, l, T, pf, pi):
    t = pf[F_TICK]
    if sd > 0:
        return h > T if not pi[P_HTICK] else h >= T + t - t * EPS
    return l < T if not pi[P_HTICK] else l <= T - t + t * EPS


@njit(cache=True)
def _stop_hit(sd, h, l, S, pi):
    if sd > 0:
        return l <= S if pi[P_TOUCH] else l < S
    return h >= S if pi[P_TOUCH] else h > S


@njit(cache=True)
def _take_level(sd, px, sg, ta, pf, pi):
    if pi[P_ARANG]:                                     # a = a0 + a1·ранг дня сигнала; нет ранга (nan) — тейка нет
        if np.isnan(ta):
            return np.nan
        m = ta * sg
    elif pi[P_HR]:
        if not pi[P_HB]:
            return np.nan
        m = pf[F_R] * pf[F_B] * math.sqrt(pf[F_H] if pi[P_HH] else float(pi[P_TAU])) * sg
    elif pi[P_HA]:
        m = pf[F_A] * sg
    else:
        return np.nan
    T = px * (1 + sd * m)
    if not pi[P_HTICK]:
        return T
    return _ceil_t(T, pf[F_TICK]) if sd > 0 else _floor_t(T, pf[F_TICK])


@njit(cache=True)
def _sg(base, st, j, arr, ix, pi):
    """Run._sg: base — единица сделки на входе (SG стопа / SGT тейка), arr — её ряд (σ̂ или u Л5) для ezhedn."""
    sg = base
    if pi[P_EZH] and j > st[EJ]:
        v = arr[ix[j - 1]]
        if np.isfinite(v):
            sg = v
    return sg if (np.isfinite(sg) and sg > 0) else np.nan


@njit(cache=True)
def _chand(sd, j, ix, B, off, nbar, atr, pf):
    """Run._chand: экстремум 11 торговых дней ДО дня j ∓ k·ATR(14) на конец дня j−1; нет окна или ATR — nan."""
    if j == 0 or not np.isfinite(atr[ix[j - 1]]):
        return np.nan
    a, ex = atr[ix[j - 1]], np.nan
    for q in range(max(0, j - CH_OKNO), j):
        k0, nb = off[ix[q]], nbar[ix[q]]
        m = B[1, k0:k0 + nb].max() if sd > 0 else B[2, k0:k0 + nb].min()
        if np.isnan(ex) or ((m > ex) if sd > 0 else (m < ex)):
            ex = m
    return ex - sd * pf[F_CH] * a


@njit(cache=True)
def _levels(fs, st, j, su, tu, ix, pf, pi, B, off, nbar, atr):
    """(S, T), nan — уровня нет; st[SR] — причина стопа (stop | trail | bu), как Run._levels (кандидаты — порядок stop_level)."""
    sg = _sg(fs[SG], st, j, su, ix, pi)
    if np.isnan(sg):
        return np.nan, np.nan
    sd, px, S, r = st[SD], fs[EPX], np.nan, 0
    if pi[P_HB]:
        S, r = px * (1 - sd * pf[F_B] * sg), R_STOP
        if pi[P_HKTR]:
            x = fs[EXT] * (1 - sd * pf[F_KTR] * sg)
            if (x > S) if sd > 0 else (x < S):
                S, r = x, R_TRAIL
    if pi[P_ATRM] and j > 0 and np.isfinite(atr[ix[j - 1]]):     # Ж-092: экстремум ∓ m·ATR(14, дни до j); нет b — единственный кандидат
        x = fs[EXT] - sd * pf[F_ATRM] * atr[ix[j - 1]]
        if np.isnan(S) or ((x > S) if sd > 0 else (x < S)):
            S, r = x, R_TRAIL
    if pi[P_HB]:
        if st[BE]:
            if (px > S) if sd > 0 else (px < S):
                S, r = px, R_BU
        if pi[P_SP] or pi[P_STEP]:                      # V6-b v6_extra: «стоп в плюс», ступенчатый — после bu, до chand; храповик vlev
            fs[VLEV] = v6.extra_level(sd, px, sg, fs[EXT], pf[F_B], st[PQ], fs[VLEV], pf, pi)
            if not np.isnan(fs[VLEV]) and ((fs[VLEV] > S) if sd > 0 else (fs[VLEV] < S)):
                S, r = fs[VLEV], R_TRAIL
        if pi[P_CH]:                                    # Л5: Chandelier — кандидат extra, затем храповик (ratchet)
            x = _chand(sd, j, ix, B, off, nbar, atr, pf)
            if not np.isnan(x) and ((x > S) if sd > 0 else (x < S)):
                S, r = x, R_TRAIL
    if pi[P_CH] or pi[P_ATRM]:                          # храповик: стоп не ослабляется; нет уровня — храповик сброшен (ratchet(None))
        if np.isnan(S):
            fs[SMAX] = np.nan
        else:
            if not np.isnan(fs[SMAX]):
                S = max(S, fs[SMAX]) if sd > 0 else min(S, fs[SMAX])
            fs[SMAX] = S
    if pi[P_PARAB]:                                     # П3b: в баре входа — начальный стоп; далее по состоянию на конец прошлого бара
        S, r = pb.stop(fs, st, sd), R_TRAIL
    if not np.isnan(S):
        st[SR] = r
        if fs[SH] != 0.0:
            S = S - fs[SH]
    st_ = _sg(fs[SGT], st, j, tu, ix, pi)
    T = np.nan if (st[XO] or np.isnan(st_)) else _take_level(sd, px, st_, fs[TA], pf, pi)
    return S, T


@njit(cache=True)
def _levels_k(fs, st, j, su, tu, ix, pf, pi, B, off, nbar, atr, UR):
    """RunStop._levels: к уровням σ добавляется стоп-заявка выхода по каналу (xl / xs прошлого торгового дня), строже из двух."""
    S, T = _levels(fs, st, j, su, tu, ix, pf, pi, B, off, nbar, atr)
    if pi[P_SZ] and j > 0:
        sd = st[SD]
        X = UR[2 if sd > 0 else 3, ix[j - 1]]
        if np.isfinite(X):
            if fs[SH] != 0.0:
                X = X - fs[SH]
            if np.isnan(S) or ((X > S) if sd > 0 else (X < S)):
                S, st[SR] = X, R_KANAL
    return S, T


@njit(cache=True)
def _row(i, j, at_close, ix):
    return i if (at_close or j == 0) else ix[j - 1]


@njit(cache=True)
def _enter(fs, st, ti, tf, tx, acc, i, j, b, px, i0, sg, d_sig, sd, nb_i, pf, pi, ix, su, tu, UR, B, off):
    n = st[NT]
    st[NT], st[CUR] = n + 1, n
    row0 = _row(i, j, i0 >= nb_i, ix)
    ti[n, DS], ti[n, ED], ti[n, EB], ti[n, SDC], ti[n, R0], ti[n, PAR] = d_sig, i, b, sd, row0, -1
    tf[n, 0] = px
    tx[n, 0], tx[n, 1] = 1.0, 0.0
    acc[1, row0] += pf[F_COMM]
    fs[REF], fs[EXT], fs[SH], fs[EPX], fs[SMAX] = px, px, 0.0, px, np.nan
    fs[SG] = su[d_sig] if pi[P_US] else sg              # Run._enter: единица Л5 — u[d_sig], иначе переданная σ̂
    fs[SGT] = tu[d_sig] if pi[P_UT] else sg
    fs[W], fs[RR] = 1.0, px * pf[F_B] * fs[SG] if pi[P_HB] else np.nan
    st[TR], st[SD], st[I0], st[EJ], st[BE], st[PART], st[ENTD], st[ENTB] = 1, sd, i0, j, 0, 0, i, b
    fs[BEM], fs[TA], fs[VLEV], st[PQ] = UR[4, d_sig], UR[5, d_sig], np.nan, 0     # v6_enter: ранг дня сигнала d_sig, не дня входа
    st[NAD], st[AD0], st[AD1] = 0, -1, -1
    if pi[P_PARAB]:                                     # П3b: экстремум на входе — high / low бара входа
        pb.start(fs, st, sd, px, pf[F_B], fs[SG], B[1, off[i] + b], B[2, off[i] + b], pf[F_PS])


@njit(cache=True)
def _exit(fs, st, ti, tf, tx, acc, i, j, b, px, reason, pf, pi, ix):
    n, sd = st[CUR], st[SD]
    ti[n, XD], ti[n, XB], ti[n, RS] = i, b, reason
    tot = 0.0                                           # v6_exit: ноги добавок закрываются вместе с родителем, своя комиссия выхода
    for k in (st[AD0], st[AD1]):
        if k >= 0:
            ti[k, XD], ti[k, XB], tf[k, 1], tx[k, 1] = i, b, px, pf[F_COMM] * tx[k, 0]
            tot += tx[k, 0]
    tf[n, 1], tx[n, 1] = px, pf[F_COMM] * (fs[W] - tot)
    acc[0, i] += sd * fs[W] * math.log((px + fs[DV]) / fs[REF])
    fs[DV] = 0.0
    acc[1, _row(i, j, pi[P_ZAK] != 0 or reason == R_VREMYA or reason == R_SIGNAL, ix)] += pf[F_COMM] * fs[W]
    acc[2, i] += (b + 1 - st[I0]) * fs[W]
    st[LXD], st[LXB], st[LXR] = i, b, reason
    if not pi[P_REENTRY]:
        st[BLK] = sd
    st[TR] = 0


@njit(cache=True)
def _partial(fs, st, ti, tf, tx, acc, i, j, b, px, pf, ix):
    """Run._partial: часть позиции лимитом на +r·R — отдельная сделка "part" (parent — индекс родителя); остаток живёт дальше."""
    n, c, sd, fr = st[NT], st[CUR], st[SD], pf[F_CF]
    st[NT] = n + 1
    ti[n, DS], ti[n, ED], ti[n, EB], ti[n, SDC], ti[n, R0], ti[n, PAR] = ti[c, DS], ti[c, ED], ti[c, EB], sd, ti[c, R0], c
    ti[n, XD], ti[n, XB], ti[n, RS] = i, b, R_PART
    tf[n, 0], tf[n, 1] = tf[c, 0], px
    tx[n, 0], tx[n, 1] = fr, pf[F_COMM] * fr
    acc[0, i] += sd * fr * math.log((px + fs[DV]) / fs[REF])       # dv достаётся и закрытой, и оставшейся половине (не обнуляется)
    acc[1, _row(i, j, False, ix)] += pf[F_COMM] * fr
    acc[2, i] += (b + 1 - st[I0]) * fs[W]
    st[I0], fs[W], st[PART] = b + 1, fs[W] - fr, 1
    tx[c, 0] = 1.0 - fr


@njit(cache=True)
def _after_bar(fs, st, j, hi, lo, su, ix, pf, pi):
    sd, px = st[SD], fs[EPX]
    if pi[P_PARAB]:
        pb.after(fs, sd, hi, lo, pf[F_PS], pf[F_PM])
    fs[EXT] = max(fs[EXT], hi) if sd > 0 else min(fs[EXT], lo)
    sg = _sg(fs[SG], st, j, su, ix, pi)
    if np.isnan(sg):
        return
    if pi[P_SP] and not st[PQ]:                         # v6_after: проход +p·σ включает «стоп в плюс»
        st[PQ] = v6.pq_reached(sd, px, sg, hi, lo, pf)
    if not st[BE] and not np.isnan(fs[BEM]):            # be (1σ), b0, b0_c·ранг; nan — порог недостижим
        m = fs[BEM] * sg
        st[BE] = (hi >= px * (1 + m)) if sd > 0 else (lo <= px * (1 - m))


@njit(cache=True)
def _chast_bar(fs, st, ti, tf, tx, acc, i, j, b, o, h, l, pf, pi, ix):
    """Run._scan: частичный выход после стопа и до тейка, строго после бара входа: лимит на P_вх + sd·R·r (проход на тик)."""
    sd = st[SD]
    if pi[P_CHAST] and not st[PART] and (i > st[ENTD] or (i == st[ENTD] and b > st[ENTB])):
        P = fs[EPX] + sd * (fs[RR] * pf[F_CR])
        if pi[P_HTICK] and pf[F_TICK] != 0.0 and not np.isfinite(P):      # эталон: round_tick(nan) — ValueError (σ̂ входа нечисловая)
            raise ValueError("cannot convert float NaN to integer")
        if pi[P_HTICK]:
            P = _ceil_t(P, pf[F_TICK]) if sd > 0 else _floor_t(P, pf[F_TICK])
        if _take_hit(sd, h, l, P, pf, pi):
            _partial(fs, st, ti, tf, tx, acc, i, j, b, max(o, P) if sd > 0 else min(o, P), pf, ix)


@njit(cache=True)
def _pir_bar(fs, st, ti, tf, tx, acc, i, j, b, o, h, l, S, sg, pf, pi, ix):
    """V6.v6_pir: добавка по тренду — проход сквозь +m·σ (m = n_add + 1), цена max(open, уровень); одна на бар, строго после бара входа,
    нужен стоп S; размер урезан по риску ≤ 2ρ (v6.pir_size). Добавка — отдельная нога "pir" (parent — сделка), своя комиссия входа."""
    sd, m, c = st[SD], st[NAD], st[CUR]
    if np.isnan(S) or m >= v6.PIR_MAX or not (i > st[ENTD] or (i == st[ENTD] and b > st[ENTB])):
        return
    L = fs[EPX] * (1 + sd * (m + 1) * sg)
    if not _take_hit(sd, h, l, L, pf, pi):
        return
    px = max(o, L) if sd > 0 else min(o, L)
    k1 = st[AD0]
    x1, a1 = (tf[k1, 0], tx[k1, 0]) if m >= 1 else (np.nan, 0.0)
    a = v6.pir_size(sd, px, S, fs[EPX], fs[W] - (0.0 + a1), x1, a1, m, fs[RR], pf[F_PIR])
    if not a > 1e-12:
        return
    n = st[NT]
    st[NT] = n + 1
    row = _row(i, j, False, ix)
    acc[0, i] += sd * fs[W] * math.log((px + fs[DV]) / fs[REF])      # прежняя позиция до цены добавки; D — ей одной
    fs[DV] = 0.0
    acc[2, i] += (b + 1 - st[I0]) * fs[W]
    acc[1, row] += pf[F_COMM] * a
    ti[n, DS], ti[n, ED], ti[n, EB], ti[n, SDC], ti[n, R0], ti[n, PAR], ti[n, RS] = ti[c, DS], i, b, sd, row, c, R_PIR
    tf[n, 0] = px
    tx[n, 0], tx[n, 1] = a, 0.0
    st[AD1 if m else AD0] = n
    fs[REF], st[I0], fs[W], st[NAD] = px, b + 1, fs[W] + a, m + 1


@njit(cache=True)
def _scan(fs, st, ti, tf, tx, acc, i, j, b0, ne, B, off, nbar, cl, su, tu, atr, UR, ix, pf, pi):
    """Run._scan / _scan_close: стоп раньше тейка, гэп — по open; zakrytie — одна проверка по close Н6. Бары b0 … ne−1."""
    sd, nb = st[SD], nbar[i]
    if pi[P_ZAK]:
        C = cl[i]
        S, T = _levels_k(fs, st, j, su, tu, ix, pf, pi, B, off, nbar, atr, UR)
        if not np.isnan(S) and _stop_hit(sd, C, C, S, pi):
            _exit(fs, st, ti, tf, tx, acc, i, j, nb - 1, C, st[SR], pf, pi, ix)
        elif not np.isnan(T) and _take_hit(sd, C, C, T, pf, pi):
            _exit(fs, st, ti, tf, tx, acc, i, j, nb - 1, C, R_TAKE, pf, pi, ix)
        else:
            _after_bar(fs, st, j, C, C, su, ix, pf, pi)
        return
    st[pb.PBN] = 0                                      # П3b: бар входа пройден
    dyn = (pi[P_PARAB] != 0 or pi[P_HKTR] != 0 or pi[P_BE] != 0 or pi[P_ATRM] != 0 or pi[P_B0] != 0 or pi[P_B0C] != 0
           or pi[P_SP] != 0 or pi[P_STEP] != 0 or pi[P_PIR] != 0)             # = Run._scan (v6 включая): chand в dyn не входит (уровень дня)
    S, T = _levels_k(fs, st, j, su, tu, ix, pf, pi, B, off, nbar, atr, UR)
    for b in range(b0, ne):
        k = off[i] + b
        o, h, l = B[0, k], B[1, k], B[2, k]
        if not np.isnan(S) and _stop_hit(sd, h, l, S, pi):
            _exit(fs, st, ti, tf, tx, acc, i, j, b, min(o, S) if sd > 0 else max(o, S), st[SR], pf, pi, ix)
            return
        _chast_bar(fs, st, ti, tf, tx, acc, i, j, b, o, h, l, pf, pi, ix)
        if not np.isnan(T) and _take_hit(sd, h, l, T, pf, pi):
            _exit(fs, st, ti, tf, tx, acc, i, j, b, max(o, T) if sd > 0 else min(o, T), R_TAKE, pf, pi, ix)
            return
        if st[XO] and _limit_hit(-sd, h, l, fs[XX], pf, pi):
            st[XF] += 1
            _exit(fs, st, ti, tf, tx, acc, i, j, b, min(o, fs[XX]) if -sd > 0 else max(o, fs[XX]), R_LIM, pf, pi, ix)
            return
        if pi[P_PIR]:                                   # V6-b: добавка после стопа, тейка и выхода-заявки (худший порядок)
            sgp = _sg(fs[SG], st, j, su, ix, pi)
            if not np.isnan(sgp):
                _pir_bar(fs, st, ti, tf, tx, acc, i, j, b, o, h, l, S, sgp, pf, pi, ix)
        if dyn:
            _after_bar(fs, st, j, h, l, su, ix, pf, pi)
            S, T = _levels_k(fs, st, j, su, tu, ix, pf, pi, B, off, nbar, atr, UR)


@njit(cache=True)
def _scan_otlozh(fs, st, ti, tf, tx, acc, i, j, B, off, nbar, cl, su, tu, atr, UR, ix, pf, pi, cut):
    """Run2._scan при отложенном выходе (t_vyhod = open2, Ж-086 п. 4): стоп и тейк — строго по барам ДО бара cut = bar_at(10:00);
    выход по сигналу — по open бара cut, комиссия — в строку начала интервала (как внутридневной), причина signal, x_t += 1.
    cut = −1 — времени баров нет: полный скан дня и выход по close Н6; cut = nb — после 10:00 баров нет: перенос на следующий день."""
    st[PEND] = 0
    nb, c = nbar[i], cut[i]
    if c < 0:
        _scan(fs, st, ti, tf, tx, acc, i, j, 0, nb, B, off, nbar, cl, su, tu, atr, UR, ix, pf, pi)
        if st[TR]:
            _exit(fs, st, ti, tf, tx, acc, i, j, nb - 1, cl[i], R_SIGNAL, pf, pi, ix)
        return
    if c > 0:
        _scan(fs, st, ti, tf, tx, acc, i, j, 0, c, B, off, nbar, cl, su, tu, atr, UR, ix, pf, pi)
    if not st[TR]:
        return
    if c < nb:
        st[XT] += 1
        _exit(fs, st, ti, tf, tx, acc, i, j, c, B[0, off[i] + c], R_STOP, pf, pi, ix)    # R_STOP: комиссия в строку начала интервала
        ti[st[CUR], RS], st[LXR] = R_SIGNAL, R_SIGNAL
    elif j + 1 < len(ix):
        st[PEND] = 1
    else:
        _exit(fs, st, ti, tf, tx, acc, i, j, nb - 1, cl[i], R_SIGNAL, pf, pi, ix)


@njit(cache=True)
def _place(fs, st, close, sd, sg, j, i, retry, age, pf, pi):
    st[NO] += 1
    fs[OL], fs[OSGO] = _limit_level(sd, close, sg, pf, pi), sg
    st[OK], st[ODUE], st[OSD], st[ODS], st[ORT], st[OAGE] = 1, j + 1, sd, i, retry, age


@njit(cache=True)
def _vhod(fs, st, ti, tf, tx, acc, i, j, b, px, sg, d_sig, sd, B, off, nbar, cl, su, tu, atr, UR, ix, pf, pi):
    """Вход в баре b по px, затем стоп в том же баре (Run._stop_vhoda, БЛОКЕР #596: «уже пробитый» выходит по px, не по S), затем скан."""
    nb, k0 = nbar[i], off[i]
    _enter(fs, st, ti, tf, tx, acc, i, j, b, px, b, sg, d_sig, sd, nb, pf, pi, ix, su, tu, UR, B, off)
    S, _ = _levels(fs, st, j, su, tu, ix, pf, pi, B, off, nbar, atr)      # в баре входа — только защитный стоп σ (Run2._levels)
    if not pi[P_ZAK] and not np.isnan(S) and _stop_hit(sd, B[1, k0 + b], B[2, k0 + b], S, pi):
        _exit(fs, st, ti, tf, tx, acc, i, j, b, min(px, S) if sd > 0 else max(px, S), st[SR], pf, pi, ix)
    else:
        _scan(fs, st, ti, tf, tx, acc, i, j, b + 1, nb, B, off, nbar, cl, su, tu, atr, UR, ix, pf, pi)


@njit(cache=True)
def _try_order_sz(fs, st, ti, tf, tx, acc, i, j, B, off, nbar, cl, sigma, su, tu, atr, UR, ix, pf, pi):
    """RunStop._try_order: OCO buy-stop up / sell-stop dn (касание, цена — худшая из open и уровня; оба в баре — ближайший к open)."""
    st[OK] = 0
    _oco(fs, st, ti, tf, tx, acc, i, j, 0, fs[OUP], fs[ODN], st[ODS], B, off, nbar, cl, sigma, su, tu, atr, UR, ix, pf, pi)


@njit(cache=True)
def _oco(fs, st, ti, tf, tx, acc, i, j, b0, up, dn, d_sig, B, off, nbar, cl, sigma, su, tu, atr, UR, ix, pf, pi):
    """exec_e_rules.stop_vhod с бара b0: первый бар с касанием, цена — худшая из open и уровня, оба — ближайший к open. Вход d_sig."""
    k0 = off[i]
    for b in range(b0, nbar[i]):
        o, h, l = B[0, k0 + b], B[1, k0 + b], B[2, k0 + b]
        buy, sell = h >= up, l <= dn
        if buy or sell:
            if buy and sell:
                buy = abs(o - up) <= abs(o - dn)
            st[NF] += 1
            px, sd = (max(o, up), 1) if buy else (min(o, dn), -1)
            _vhod(fs, st, ti, tf, tx, acc, i, j, b, px, sigma[d_sig], d_sig, sd, B, off, nbar, cl, su, tu, atr, UR, ix, pf, pi)
            return True
    return False


@njit(cache=True)
def _reenter_sz(fs, st, ti, tf, tx, acc, i, j, B, off, nbar, cl, sigma, su, tu, atr, UR, ix, pf, pi):
    """RunStop._reenter (V6-11c): после внутридневного выхода дня i пара вечера j−1 стоит на барах СТРОГО после бара выхода; ≤ 1 за день."""
    r = st[LXR]
    if st[TR] or st[OK] or j == 0 or st[LXD] != i or r == R_VREMYA or r == R_SIGNAL or st[DNRE] == i or st[LXB] + 1 >= nbar[i]:
        return
    st[DNRE] = i
    ip = ix[j - 1]
    if not np.isfinite(sigma[ip]):
        return
    up, dn = UR[0, ip], UR[1, ip]
    if pf[F_KE] != 0.0:
        a = pf[F_KE] * atr[ip]
        up, dn = up + a, dn - a
    if not (np.isfinite(up) or np.isfinite(dn)):
        return
    up, dn = (up if np.isfinite(up) else np.inf), (dn if np.isfinite(dn) else -np.inf)
    if _oco(fs, st, ti, tf, tx, acc, i, j, st[LXB] + 1, up, dn, ip, B, off, nbar, cl, sigma, su, tu, atr, UR, ix, pf, pi):
        st[NO] += 1


@njit(cache=True)
def _reenter(fs, st, ti, tf, tx, acc, i, j, B, off, nbar, cl, dirs, sigma, su, tu, atr, UR, ix, pf, pi):
    """exec_e_zayavki.reenter: после внутридневного выхода дня i лимит вечера j−1 стоит на барах СТРОГО после бара выхода; ≤ 1 за день."""
    nb, r = nbar[i], st[LXR]
    if st[TR] or st[OK] or j == 0 or st[LXD] != i or r == R_VREMYA or r == R_SIGNAL or st[DNRE] == i or st[LXB] + 1 >= nb:
        return
    ip = ix[j - 1]
    sd, sg = dirs[ip], sigma[ip]
    if sd == 0 or not np.isfinite(sg):
        return
    st[DNRE] = i
    _place(fs, st, cl[ip], sd, sg, j - 1, ip, 0, 1, pf, pi)
    st[OK] = 0
    k0 = off[i]
    for b in range(st[LXB] + 1, nb):
        if _limit_hit(sd, B[1, k0 + b], B[2, k0 + b], fs[OL], pf, pi):
            o = B[0, k0 + b]
            st[NF] += 1
            _vhod(fs, st, ti, tf, tx, acc, i, j, b, min(o, fs[OL]) if sd > 0 else max(o, fs[OL]), sg, ip, sd, B, off, nbar, cl,
                  su, tu, atr, UR, ix, pf, pi)
            return


@njit(cache=True)
def _try_order(fs, st, ti, tf, tx, acc, i, j, B, off, nbar, cl, dirs, sigma, su, tu, atr, UR, ix, pf, pi):
    """Run._try_order: первый бар с проходом на тик; цена — L или open при гэпе; затем стоп в том же баре."""
    if pi[P_SZ]:
        return _try_order_sz(fs, st, ti, tf, tx, acc, i, j, B, off, nbar, cl, sigma, su, tu, atr, UR, ix, pf, pi)
    st[OK] = 0
    sd, L, nb, k0 = st[OSD], fs[OL], nbar[i], off[i]
    for b in range(nb):
        if _limit_hit(sd, B[1, k0 + b], B[2, k0 + b], L, pf, pi):
            o = B[0, k0] if pi[P_ZAK] else B[0, k0 + b]
            px = min(o, L) if sd > 0 else max(o, L)
            st[NF] += 1
            sg = sigma[st[ODS]] if not st[ORT] else fs[OSGO]
            _vhod(fs, st, ti, tf, tx, acc, i, j, b, px, sg, st[ODS], sd, B, off, nbar, cl, su, tu, atr, UR, ix, pf, pi)
            return
    close, live = cl[i], dirs[i] == sd
    if (not st[ORT] and st[OAGE] < pi[P_T] and live and j + 1 < len(ix) and np.isfinite(sigma[i])):   # V6-a prodlit: заявка живёт, пока сигнал активен
        _place(fs, st, close, sd, sigma[i], j, i, 0, st[OAGE] + 1, pf, pi)
        return
    if st[ORT] or pi[P_F] == 0:
        st[BLK] = sd
    elif pi[P_F] == 1:
        if live:
            _enter(fs, st, ti, tf, tx, acc, i, j, nb - 1, close, nb, sigma[i], st[ODS], sd, nb, pf, pi, ix, su, tu, UR, B, off)
    elif pi[P_F] == 2:
        v = sigma[i]
        if live and np.isfinite(v) and j + 1 < len(ix):
            _place(fs, st, close, sd, v, j, i, 1, 1, pf, pi)
        else:
            st[BLK] = sd


@njit(cache=True)
def _atr(B, off, nbar, pusto, cl):
    """= exec_e_rules.atr_series(days, close): a[i] на close дня i, сумма последних 14 TR слева направо / 14; pusto — перенос."""
    n = len(nbar)
    a, tr = np.full(n, np.nan), np.empty(n)
    c, pc = 0, np.nan
    for i in range(n):
        if pusto[i]:
            a[i] = a[i - 1] if i else np.nan
            continue
        k0, nb = off[i], nbar[i]
        h, lo = B[1, k0:k0 + nb].max(), B[2, k0:k0 + nb].min()
        tr[c] = h - lo if c == 0 else max(h - lo, abs(h - pc), abs(lo - pc))
        c += 1
        pc = cl[i]
        if c >= ATR_N:
            s = 0.0
            for q in range(c - ATR_N, c):
                s += tr[q]
            a[i] = s / ATR_N
    return a


@njit(cache=True)
def _unit(kind, atr, cl, pusto):
    """= exec_e.unit_series: pct — 1; atr — ATR / close последнего торгового дня (≤ i)."""
    n = len(pusto)
    if kind == 2:
        return np.ones(n)
    u, last = np.empty(n), np.nan
    for i in range(n):
        if not pusto[i]:
            last = cl[i]
        u[i] = atr[i] / last
    return u


@njit(cache=True)
def kernel(B, off, nbar, pusto, cl, dirs, sigma, div, pf, pi, stale):
    """Прежний вход ядра (fast_etap1.run): pf[10], pi[21] — новые оси выключены; возврат (gross, cost, pos, free, ti[8], tf, st)."""
    pf2, pi2 = np.zeros(24), np.zeros(34, np.int64)
    pf2[:len(pf)], pi2[:len(pi)], pi2[P_T] = pf, pi, 1
    g, c, p, f, ti, tf, _, st = kernel_v5(B, off, nbar, pusto, cl, dirs, sigma, div, pf2, pi2, np.empty((4, 0)), np.empty(0),
                                          np.empty(0, np.int64), stale)
    return g, c, p, f, np.ascontiguousarray(ti[:, :8]), tf, st


@njit(cache=True)
def kernel_v5(B, off, nbar, pusto, cl, dirs, sigma, div, pf, pi, UR0, rang, cut, stale):  # xtask: длинная — день Run.step целиком, порядок важен
    """Run.step по всем торговым дням + постобработка exec_e.run. UR0[4, дни] — up, dn, xl, xs (stop_zayavka; иначе (4, 0));
    rang[дни] — ранг силы на день сигнала (оси b0_c, a_rang; иначе пустой). cut[дни] — бар 10:00 дня (exec_e_vremya.bar_at; −1 — времени
    нет; нужен при pi[P_TV], иначе пустой). stale[j] — разрыв > razryv дней перед шагом j (Д5-2; пустой — нет). Внутри UR = UR0 + строки порога безубытка и a тейка.
    Возврат: (gross, cost, pos, free, ti[9], tf[2: вход, выход], tx[2: доля, комиссия выхода], st)."""
    n = len(nbar)
    UR = np.full((6, n), np.nan)
    if UR0.shape[1]:
        UR[:4] = UR0
    UR[4], UR[5] = v6.rows(rang, n, pf, pi, pi[P_BE])
    ix = np.flatnonzero(~pusto)
    nix = len(ix)
    need = pi[P_CH] != 0 or pi[P_US] == 1 or pi[P_UT] == 1 or pi[P_ATRM] != 0 or pf[F_KE] != 0.0
    atr = _atr(B, off, nbar, pusto, cl) if need else np.empty(0)
    su = _unit(pi[P_US], atr, cl, pusto) if pi[P_US] else sigma
    tu = _unit(pi[P_UT], atr, cl, pusto) if pi[P_UT] else sigma
    fs, st = np.zeros(21), np.zeros(NST, np.int64)
    st[LXD], st[DNRE] = -1, -1
    cap = 3 * (1 + (pi[P_CHAST] != 0) + 2 * (pi[P_PIR] != 0)) * nix + 4   # входов за день ≤ 3 (заявка, повтор, close), у каждого ≤ 1 частичный выход и ≤ 2 добавок
    ti, tf, tx = np.full((cap, 9), -1, np.int64), np.zeros((cap, 2)), np.zeros((cap, 2))
    acc, pos = np.zeros((3, n)), np.zeros(n)                # gross, cost, used по дням
    for j in range(nix):
        i = ix[j]
        nb = nbar[i]
        fs[DV] = div[i] * pf[F_NAL] if st[TR] else 0.0
        if st[TR]:
            st[I0] = 0
            fs[SH] += div[i]
        if len(stale) and stale[j]:                         # Д5-2: заявка через остановку торгов снята; позиция держится (Run.step)
            st[OK] = 0
        if st[OK] and st[ODUE] == j:
            _try_order(fs, st, ti, tf, tx, acc, i, j, B, off, nbar, cl, dirs, sigma, su, tu, atr, UR, ix, pf, pi)
        elif st[TR] and st[PEND]:
            _scan_otlozh(fs, st, ti, tf, tx, acc, i, j, B, off, nbar, cl, su, tu, atr, UR, ix, pf, pi, cut)
        elif st[TR]:
            _scan(fs, st, ti, tf, tx, acc, i, j, 0, nb, B, off, nbar, cl, su, tu, atr, UR, ix, pf, pi)
        if pi[P_RED] and not (len(stale) and stale[j]):     # V6-a: повторный вход в тот же день после внутридневного выхода
            if pi[P_SZ]:                                    # на стоп-заявке повторный вход — та же OCO-пара (RunStop._reenter)
                _reenter_sz(fs, st, ti, tf, tx, acc, i, j, B, off, nbar, cl, sigma, su, tu, atr, UR, ix, pf, pi)
            else:
                _reenter(fs, st, ti, tf, tx, acc, i, j, B, off, nbar, cl, dirs, sigma, su, tu, atr, UR, ix, pf, pi)
        close = cl[i]
        if st[XO]:
            if st[TR]:
                _exit(fs, st, ti, tf, tx, acc, i, j, nb - 1, close, st[XR], pf, pi, ix)
            st[XO] = 0
        if st[TR]:
            lv = 0
            if pi[P_HTAU] and j - st[EJ] >= pi[P_TAU] and j > st[EJ]:
                lv = R_VREMYA
            elif pi[P_EXS] and dirs[i] != st[SD]:
                lv = R_SIGNAL
            if lv and st[PEND]:                             # Run2._leave: выход уже перенесён на open следующего дня
                lv = 0
            if lv == R_SIGNAL and pi[P_TV] and j + 1 < nix:   # open2: по сигналу — open x+1; по τ — close дня истечения (vyhod_plan)
                st[PEND] = 1
            elif lv:
                sg = sigma[i]
                if pi[P_VLIM] and j + 1 < nix and np.isfinite(sg):
                    st[XN] += 1
                    st[XO], st[XR], fs[XX] = 1, lv, _limit_level(-st[SD], close, sg, pf, pi)
                else:
                    _exit(fs, st, ti, tf, tx, acc, i, j, nb - 1, close, lv, pf, pi, ix)
        if st[TR]:
            acc[0, i] += st[SD] * fs[W] * math.log((close + fs[DV]) / fs[REF])
            fs[DV], fs[REF] = 0.0, close
            acc[2, i] += (nb - st[I0]) * fs[W]
            st[I0] = nb
        if st[BLK] and dirs[i] != st[BLK]:
            st[BLK] = 0
        sg, sd = sigma[i], dirs[i]
        if pi[P_SZ]:                                        # RunStop._decide: OCO-пара стоп-заявок на d+1, сигнал и блок не смотрятся
            if not (st[TR] or st[OK] or j + 1 >= nix or not np.isfinite(sg)):
                up, dn = UR[0, i], UR[1, i]
                if pf[F_KE] != 0.0:                         # Л6 ke_levels: сдвиг на k_e·ATR(close d); нет ATR — заявки нет
                    a = pf[F_KE] * atr[i]
                    up, dn = up + a, dn - a
                if np.isfinite(up) or np.isfinite(dn):
                    st[NO] += 1
                    fs[OUP], fs[ODN] = (up if np.isfinite(up) else np.inf), (dn if np.isfinite(dn) else -np.inf)
                    st[OK], st[ODUE], st[ODS] = 1, j + 1, i
        elif not (st[TR] or st[OK] or st[BLK] or not sd or not np.isfinite(sg)):
            if not pi[P_HK]:
                _enter(fs, st, ti, tf, tx, acc, i, j, nb - 1, close, nb, sg, i, sd, nb, pf, pi, ix, su, tu, UR, B, off)
            elif j + 1 < nix:
                _place(fs, st, close, sd, sg, j, i, 0, 1, pf, pi)
        pos[i] = float(st[SD]) * fs[W] if st[TR] else 0.0
    if nix and st[TR]:
        acc[1, ix[-1]] += pf[F_COMM] * fs[W]
    gross, free = np.zeros(n), np.zeros(n)
    for j in range(1, nix):
        gross[ix[j - 1]] += acc[0, ix[j]]
        free[ix[j - 1]] = 1.0 - acc[2, ix[j]] / nbar[ix[j]]
    if nix:
        free[ix[-1]] = 1.0 - abs(pos[ix[-1]])
    for i in range(n):
        if pusto[i]:
            pos[i] = pos[i - 1] if i else 0.0
    return gross, acc[1].copy(), pos, free, ti[:st[NT]], tf[:st[NT]], tx[:st[NT]], st


# --- обёртка v5: drop-in exec_e.run для осей S3/S4; fast_etap1.run их не знает (включение в pokryto — отдельный коммит) ---
NOVYE = frozenset(("razryv", "vhod_tip", "atr_m", "chast_1r", "chast", "T", "reentry_den", "k_e", "t_vyhod"))     # + v6.OSI (b0, b0_c, sp, step_s, a_rang, pir) при YADRO_OSI3_GOTOV
T_VYHOD = ("close", "open2")                            # из t_vyhod ядро знает только эти (1500 / 1700 — эталон)


def ne_znaet(p) -> list:
    """Поля Params не в умолчании, которых нет ни в FAST_POLYA, ни в осях v5 ядра (пусто — ядро режим знает)."""
    from fastbt import fast_etap1
    sz = p.vhod_tip == "stop_zayavka"           # Ф-vhod_open: на стоп-заявке vhod_open ничего не меняет (RunStop._decide не смотрит на него)
    bad = [n for n, d in fast_etap1._umolch(type(p)) if n not in NOVYE and not (YADRO_OSI3_GOTOV and n in v6.OSI)
           and not (YADRO_PARAB_GOTOV and n == "parab") and getattr(p, n) != d and not (sz and n == "vhod_open")]
    return bad + ["t_vyhod"] * (p.t_vyhod not in T_VYHOD)


def params_v5(p, nal: float) -> tuple:
    from fastbt import fast_etap1
    from fastbt.exec_e_v6 import chast_pair
    pf, pi = fast_etap1._params(p, nal)
    cf = chast_pair(p)
    pf = np.r_[pf, [0.0 if not p.k_e else p.k_e, p.atr_m or 0.0, cf[0] if cf else 0.0, cf[1] if cf else 0.0]]
    pi = np.r_[pi, [p.vhod_tip == "stop_zayavka", p.atr_m is not None, p.T, p.reentry_den, cf is not None]].astype(np.int64)
    f6, i6 = v6.params_v6(p)
    fp, ip = pb.params_parab(p)
    return np.r_[pf, f6, fp], np.r_[pi, i6, [p.t_vyhod == "open2"], ip].astype(np.int64)


def stale_mask(dates, pusto, razryv) -> np.ndarray:
    """stale[j] по торговым шагам j (как `exec_e.stale_steps`): разрыв с предыдущим торговым днём > razryv календарных дней. Нет правила — пустой."""
    ix = np.flatnonzero(~pusto)
    if razryv is None or dates is None or len(ix) < 2:
        return np.empty(0, np.bool_)
    d = pd.DatetimeIndex(dates)[ix]
    return np.r_[False, np.asarray((d[1:] - d[:-1]).days > razryv)].astype(np.bool_)


def cut_open2(days, p) -> np.ndarray:
    """Бар 10:00 каждого дня (exec_e_vremya.bar_at; −1 — времени нет или день пуст) при t_vyhod = open2, иначе пустой массив."""
    if p.t_vyhod != "open2":
        return np.empty(0, np.int64)
    from fastbt import exec_e_vremya as vr
    return np.array([-1 if d.pusto else (-1 if (b := vr.bar_at(d, vr.HHMM["open2"])) is None else b) for d in days], np.int64)


def _trades_v5(ticker, ti, tf, tx, comm) -> list:
    from fastbt.exec_e_types import Trade
    out = []
    for r, (epx, xpx), (frac, fee) in zip(ti.tolist(), tf.tolist(), tx.tolist()):
        done = r[XD] >= 0
        part = r[PAR] >= 0
        pir = part and r[RS] == R_PIR                   # нога добавки: reason "pir" с момента входа, своя комиссия входа comm·frac
        out.append(Trade(ticker, r[DS], r[ED], r[EB], epx, r[XD] if done else None, r[XB] if done else None, xpx if done else None,
                         REASONS[r[RS]] if (done or pir) else None, r[SDC], comm * frac if pir else (0.0 if part else comm),
                         fee if done else 0.0, r[R0], frac, r[PAR] if part else None))
    return out


def _v_bare(ticker, days, close, p, trades) -> None:
    """Инвариант исполнения (#610, #617; Run._v_bare): цена входа и выхода сделки — в [low; high] своего бара (zakrytie / bar_plus —
    диапазон дня); close дня (Н6) — цена рынка. Нарушение — ValueError тем же текстом, что у эталона."""
    for t in trades:
        for den, b, px in ((t.entry_day, t.entry_bar, t.entry_px), (t.exit_day, t.exit_bar, t.exit_px)):
            if den is None:
                continue
            d = days[den]
            b = min(b, len(d.c) - 1)
            lo, hi = (min(d.l), max(d.h)) if (p.check == "zakrytie" or p.bar_plus) else (d.l[b], d.h[b])
            if px != (float(close[den]) if close is not None else d.c[-1]) and not (lo - abs(lo) * 1e-9 <= px <= hi + abs(hi) * 1e-9):
                raise ValueError(f"исполнение вне [low; high] бара: {ticker} день {den} бар {b} цена {px} [{lo}; {hi}]")


def run_v5(ticker, days, sig, sigma, p, rf_cal=None, close=None, div=None, nalog=None, urovni=None, rang=None, dates=None):
    """Drop-in `exec_e.run` для точек с осями stop_zayavka+urovni, atr_m, T, reentry_den, chast_1r / chast (и осями ядра).
    Режим вне ядра — NotImplementedError (его считает эталон); те же проверки входов и тот же Result, что у эталона."""
    from fastbt import dividendy, exec_e, fast_etap1
    from fastbt.exec_e_types import Result
    sig, sigma = np.asarray(sig), np.ascontiguousarray(sigma, float)
    if (bad := ne_znaet(p)):
        raise NotImplementedError(f"ядро не знает поля Params: {bad}")
    if not len(days) or not np.isfinite(sig.astype(float)).all():
        return fast_etap1._SLOW_RUN(ticker, days, sig, sigma, p, rf_cal=rf_cal, close=close, div=div, nalog=nalog, urovni=urovni, rang=rang)
    if not (len(days) == len(sig) == len(sigma)) or (close is not None and len(close) != len(days)):
        raise ValueError("days, sig, sigma, close разной длины")
    exec_e._check_params(p)
    if (p.vhod_tip == "stop_zayavka") != (urovni is not None):
        raise ValueError("urovni нужны ровно при vhod_tip = stop_zayavka")
    if urovni is not None and any(len(urovni[k]) != len(days) for k in ("up", "dn", "xl", "xs")):
        raise ValueError("days, urovni разной длины")
    B, off, nbar, pusto, bad_d = fast_etap1.flat(days)
    cl = B[3, off + np.maximum(nbar - 1, 0)] if close is None else np.asarray(close, float)
    bad_c = ~pusto & ~(np.isfinite(cl) & (cl > 0)) if close is not None else np.zeros(len(days), bool)
    if (bad_d | bad_c).any():
        i = int(np.flatnonzero(bad_d | bad_c)[0])
        raise ValueError(f"день {i}: " + ("пустой или нечисловой набор баров (нужен pusto=True)" if bad_d[i] else "close Н6 нечисловой"))
    if div is not None and len(div) != len(days):
        raise ValueError("days, div разной длины")
    if p.razryv is not None and dates is not None and len(dates) != len(days):
        raise ValueError("dates и days разной длины")
    from fastbt.exec_e_v6 import v5_check, v6_check
    v6_check(p, rang, len(days))                        # те же отказы, что в Run.__init__ (ранг нужен b0_c / a_rang)
    v5_check(p, rang, len(days))
    ds, ix = np.zeros(len(days)), np.flatnonzero(~pusto)
    for i in np.flatnonzero(np.asarray(div, float)) if div is not None else ():
        k = np.searchsorted(ix, i)
        if k < len(ix):
            ds[ix[k]] += float(div[i])
    dirs = (p.side * np.sign(sig.astype(float))).astype(np.int64)
    pf, pi = params_v5(p, float(dividendy.DIV_NALOG if nalog is None else nalog))
    UR = np.array([urovni[k] for k in ("up", "dn", "xl", "xs")], float) if urovni is not None else np.empty((4, 0))
    gross, cost, pos, free, ti, tf, tx, st = kernel_v5(B, off, nbar, pusto, cl, dirs, sigma, ds, pf, pi, UR, np.asarray(rang, float) if rang is not None else np.empty(0),
                                                       cut_open2(days, p), stale_mask(dates, pusto, p.razryv))
    trades = _trades_v5(ticker, ti, tf, tx, p.comm)
    _v_bare(ticker, days, close, p, trades)
    rf = np.zeros(len(days)) if rf_cal is None else np.asarray(rf_cal, float)
    return Result(trades, gross, cost, pos, free, free * rf, int(st[NO]), int(st[NF]), int(st[XN]), int(st[XF]), int(st[XT]))
