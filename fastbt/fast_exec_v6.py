"""Оси V6-b в быстром ядре (Ж-083, лист 3 из П-0): b0, b0_c (порог безубытка), sp («стоп в плюс»), step_s (ступенчатый трейлинг),
a_rang (тейк от ранга силы) и вход rang. Построчно как `exec_e_v6` (be_mult, take_params, step_level, plus_level, v6_extra, v6_after);
результат побайтно равен эталону. pir (пирамидинг) здесь нет — он меняет число ног и уходит в эталон (`fast_exec.ne_znaet`).
Модуль не импортирует fast_exec (индексы осей — здесь, fast_exec их берёт), ранг дня сигнала d_sig — строки UR[4], UR[5].
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

EPS = 1e-6                                                       # = exec_e_rules.EPS
F_B0, F_B0C, F_SP, F_SQ, F_STEP, F_AA0, F_AA1, F_PIR = range(14, 22)    # pf: добавка к pf ядра v5 (14 значений)
P_B0, P_B0C, P_SP, P_STEP, P_ARANG, P_PIR = range(26, 32)               # pi: оси включены (после P_CHAST)
OSI = frozenset(("b0", "b0_c", "sp", "step_s", "a_rang", "pir"))
PIR_MAX, RISK_MAX = 2, 2.0                                            # = exec_e_v6: ≤ 2 добавок, риск ставки ≤ 2ρ


def params_v6(p) -> tuple:
    """(pf[8], pi[6]) осей листов 3a–3c; sp = (p, q)."""
    sp = p.sp if p.sp is not None else (0.0, 0.0)
    ar = p.a_rang if p.a_rang is not None else (0.0, 0.0)
    pf = [p.b0 or 0.0, p.b0_c or 0.0, sp[0], sp[1], p.step_s or 0.0, ar[0], ar[1], p.pir]
    pi = [p.b0 is not None, p.b0_c is not None, p.sp is not None, p.step_s is not None, p.a_rang is not None, p.pir > 0]
    return np.array(pf, float), np.array(pi, np.int64)


@njit(cache=True)
def rows(rang, n, pf, pi, be):
    """По дню сигнала d: порог безубытка в σ (Run.v6_enter / be_mult: b0 | b0_c·ранг | 1 при be | nan — нет) и a тейка
    (take_params: a0 + a1·ранг; нечисловой ранг — nan, тейка нет)."""
    bem, ta = np.full(n, np.nan), np.full(n, np.nan)
    for d in range(n):
        r = rang[d] if len(rang) else np.nan
        if pi[P_B0]:
            bem[d] = pf[F_B0]
        elif pi[P_B0C]:
            bem[d] = pf[F_B0C] * r if np.isfinite(r) else np.nan
        elif be:
            bem[d] = 1.0
        if pi[P_ARANG] and np.isfinite(r):
            ta[d] = pf[F_AA0] + pf[F_AA1] * r
    return bem, ta


@njit(cache=True)
def extra_level(sd, px, sg, ext, b, pq_on, vlev, pf, pi):
    """V6.v6_extra: кандидат стопа «в плюс» / ступенчатый; новый vlev = ratchet(лучший из кандидатов, прежний); nan — нет уровня."""
    best, has = np.nan, False
    if pi[P_SP] and pq_on:
        best, has = px * (1 + sd * pf[F_SQ] * sg), True
    if pi[P_STEP]:
        n = math.floor(sd * (ext - px) / (px * pf[F_STEP] * sg) + EPS)
        if n > 0:
            x = px * (1 - sd * b * sg) + sd * n * pf[F_STEP] * sg * px
            if not has or ((x > best) if sd > 0 else (x < best)):
                best = x
            has = True
    if not has:
        return vlev
    if np.isnan(vlev):
        return best
    return max(best, vlev) if sd > 0 else min(best, vlev)


@njit(cache=True)
def pir_size(sd, p_add, S, epx, w_base, x1, a1, n_add, r0, a):
    """exec_e_v6.pir_size: a, урезанный так, чтобы риск ног до стопа S не превысил 2·r0. Ноги: база (epx, w_base) и,
    если n_add = 1, добавка (x1, a1); сумма — слева направо от 0 (как sum)."""
    risk = 0.0 + w_base * max(0.0, sd * (epx - S))
    if n_add >= 1:
        risk += a1 * max(0.0, sd * (x1 - S))
    room = RISK_MAX * r0 - risk
    if room <= 0:
        return 0.0
    dr = sd * (p_add - S)
    return a if dr <= 0 else min(a, room / dr)


@njit(cache=True)
def pq_reached(sd, px, sg, hi, lo, pf):
    """V6.v6_after: проход +p·σ включает «стоп в плюс» (be_reached с множителем p)."""
    m = pf[F_SP] * sg
    return hi >= px * (1 + m) if sd > 0 else lo <= px * (1 - m)
