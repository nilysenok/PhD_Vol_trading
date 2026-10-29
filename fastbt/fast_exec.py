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
from numba import njit

EPS = 1e-6                                              # = exec_e_rules.EPS
REF, SG, EXT, SH, DV, OL, OSGO, XX, EPX, SGT, SMAX = range(11)   # fs (SGT — единица тейка, SMAX — храповик стопа)
TR, SD, I0, EJ, BE, BLK, OK, ODUE, OSD, ODS, ORT, XO, XR, SR, NO, NF, XN, XF, NT = range(19)   # st
DS, ED, EB, XD, XB, RS, SDC, R0 = range(8)              # ti
F_K, F_A, F_B, F_COMM, F_TICK, F_R, F_H, F_KTR, F_NAL, F_CH = range(10)       # pf
(P_HK, P_F, P_HA, P_HB, P_HTAU, P_TAU, P_EZH, P_EXS, P_SIDE, P_HTICK, P_TOUCH, P_HR, P_HH, P_HKTR, P_BE,
 P_ZAK, P_REENTRY, P_VLIM, P_US, P_UT, P_CH) = range(21)                     # pi (P_US/P_UT — код ED, P_CH — chand задан)
REASONS = (None, "stop", "trail", "bu", "take", "vremya", "signal", "lim")
R_STOP, R_TRAIL, R_BU, R_TAKE, R_VREMYA, R_SIGNAL, R_LIM = range(1, 8)
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
def _take_level(sd, px, sg, pf, pi):
    if pi[P_HR]:
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
    sd, px, S = st[SD], fs[EPX], np.nan
    if pi[P_HB]:
        S, r = px * (1 - sd * pf[F_B] * sg), R_STOP
        if pi[P_HKTR]:
            x = fs[EXT] * (1 - sd * pf[F_KTR] * sg)
            if (x > S) if sd > 0 else (x < S):
                S, r = x, R_TRAIL
        if st[BE]:
            if (px > S) if sd > 0 else (px < S):
                S, r = px, R_BU
        if pi[P_CH]:                                    # Л5: Chandelier — кандидат extra, затем храповик (ratchet)
            x = _chand(sd, j, ix, B, off, nbar, atr, pf)
            if not np.isnan(x) and ((x > S) if sd > 0 else (x < S)):
                S, r = x, R_TRAIL
            if not np.isnan(fs[SMAX]):
                S = max(S, fs[SMAX]) if sd > 0 else min(S, fs[SMAX])
            fs[SMAX] = S
        st[SR] = r
        if fs[SH] != 0.0:
            S = S - fs[SH]
    st_ = _sg(fs[SGT], st, j, tu, ix, pi)
    T = np.nan if (st[XO] or np.isnan(st_)) else _take_level(sd, px, st_, pf, pi)
    return S, T


@njit(cache=True)
def _row(i, j, at_close, ix):
    return i if (at_close or j == 0) else ix[j - 1]


@njit(cache=True)
def _enter(fs, st, ti, tf, acc, i, j, b, px, i0, sg, d_sig, sd, nb_i, pf, pi, ix, su, tu):
    n = st[NT]
    st[NT] = n + 1
    row0 = _row(i, j, i0 >= nb_i, ix)
    ti[n, DS], ti[n, ED], ti[n, EB], ti[n, SDC], ti[n, R0] = d_sig, i, b, sd, row0
    tf[n, 0] = px
    acc[1, row0] += pf[F_COMM]
    fs[REF], fs[EXT], fs[SH], fs[EPX], fs[SMAX] = px, px, 0.0, px, np.nan
    fs[SG] = su[d_sig] if pi[P_US] else sg              # Run._enter: единица Л5 — u[d_sig], иначе переданная σ̂
    fs[SGT] = tu[d_sig] if pi[P_UT] else sg
    st[TR], st[SD], st[I0], st[EJ], st[BE] = 1, sd, i0, j, 0


@njit(cache=True)
def _exit(fs, st, ti, tf, acc, i, j, b, px, reason, pf, pi, ix):
    n, sd = st[NT] - 1, st[SD]
    ti[n, XD], ti[n, XB], ti[n, RS] = i, b, reason
    tf[n, 1] = px
    acc[0, i] += sd * math.log((px + fs[DV]) / fs[REF])
    fs[DV] = 0.0
    acc[1, _row(i, j, pi[P_ZAK] != 0 or reason == R_VREMYA or reason == R_SIGNAL, ix)] += pf[F_COMM]
    acc[2, i] += b + 1 - st[I0]
    if not pi[P_REENTRY]:
        st[BLK] = sd
    st[TR] = 0


@njit(cache=True)
def _after_bar(fs, st, j, hi, lo, su, ix, pi):
    sd, px = st[SD], fs[EPX]
    fs[EXT] = max(fs[EXT], hi) if sd > 0 else min(fs[EXT], lo)
    sg = _sg(fs[SG], st, j, su, ix, pi)
    if pi[P_BE] and not st[BE] and not np.isnan(sg):
        st[BE] = (hi >= px * (1 + sg)) if sd > 0 else (lo <= px * (1 - sg))


@njit(cache=True)
def _scan(fs, st, ti, tf, acc, i, j, b0, B, off, nbar, cl, su, tu, atr, ix, pf, pi):
    """Run._scan / _scan_close: стоп раньше тейка, гэп — по open; zakrytie — одна проверка по close Н6."""
    sd, nb = st[SD], nbar[i]
    if pi[P_ZAK]:
        C = cl[i]
        S, T = _levels(fs, st, j, su, tu, ix, pf, pi, B, off, nbar, atr)
        if not np.isnan(S) and _stop_hit(sd, C, C, S, pi):
            _exit(fs, st, ti, tf, acc, i, j, nb - 1, C, st[SR], pf, pi, ix)
        elif not np.isnan(T) and _take_hit(sd, C, C, T, pf, pi):
            _exit(fs, st, ti, tf, acc, i, j, nb - 1, C, R_TAKE, pf, pi, ix)
        else:
            _after_bar(fs, st, j, C, C, su, ix, pi)
        return
    dyn = pi[P_HKTR] != 0 or pi[P_BE] != 0               # = Run._scan: chand в dyn не входит (уровень дня)
    S, T = _levels(fs, st, j, su, tu, ix, pf, pi, B, off, nbar, atr)
    for b in range(b0, nb):
        k = off[i] + b
        o, h, l = B[0, k], B[1, k], B[2, k]
        if not np.isnan(S) and _stop_hit(sd, h, l, S, pi):
            _exit(fs, st, ti, tf, acc, i, j, b, min(o, S) if sd > 0 else max(o, S), st[SR], pf, pi, ix)
            return
        if not np.isnan(T) and _take_hit(sd, h, l, T, pf, pi):
            _exit(fs, st, ti, tf, acc, i, j, b, max(o, T) if sd > 0 else min(o, T), R_TAKE, pf, pi, ix)
            return
        if st[XO] and _limit_hit(-sd, h, l, fs[XX], pf, pi):
            st[XF] += 1
            _exit(fs, st, ti, tf, acc, i, j, b, min(o, fs[XX]) if -sd > 0 else max(o, fs[XX]), R_LIM, pf, pi, ix)
            return
        if dyn:
            _after_bar(fs, st, j, h, l, su, ix, pi)
            S, T = _levels(fs, st, j, su, tu, ix, pf, pi, B, off, nbar, atr)


@njit(cache=True)
def _place(fs, st, close, sd, sg, j, i, retry, pf, pi):
    st[NO] += 1
    fs[OL], fs[OSGO] = _limit_level(sd, close, sg, pf, pi), sg
    st[OK], st[ODUE], st[OSD], st[ODS], st[ORT] = 1, j + 1, sd, i, retry


@njit(cache=True)
def _try_order(fs, st, ti, tf, acc, i, j, B, off, nbar, cl, dirs, sigma, su, tu, atr, ix, pf, pi):
    """Run._try_order: первый бар с проходом на тик; цена — L или open при гэпе; затем стоп в том же баре."""
    st[OK] = 0
    sd, L, nb, k0 = st[OSD], fs[OL], nbar[i], off[i]
    for b in range(nb):
        if _limit_hit(sd, B[1, k0 + b], B[2, k0 + b], L, pf, pi):
            o = B[0, k0] if pi[P_ZAK] else B[0, k0 + b]
            px = min(o, L) if sd > 0 else max(o, L)
            st[NF] += 1
            sg = sigma[st[ODS]] if not st[ORT] else fs[OSGO]
            _enter(fs, st, ti, tf, acc, i, j, b, px, b, sg, st[ODS], sd, nb, pf, pi, ix, su, tu)
            S, _ = _levels(fs, st, j, su, tu, ix, pf, pi, B, off, nbar, atr)
            if not pi[P_ZAK] and not np.isnan(S) and _stop_hit(sd, B[1, k0 + b], B[2, k0 + b], S, pi):
                _exit(fs, st, ti, tf, acc, i, j, b, S, st[SR], pf, pi, ix)
            else:
                _scan(fs, st, ti, tf, acc, i, j, b + 1, B, off, nbar, cl, su, tu, atr, ix, pf, pi)
            return
    close, live = cl[i], dirs[i] == sd
    if st[ORT] or pi[P_F] == 0:
        st[BLK] = sd
    elif pi[P_F] == 1:
        if live:
            _enter(fs, st, ti, tf, acc, i, j, nb - 1, close, nb, sigma[i], st[ODS], sd, nb, pf, pi, ix, su, tu)
    elif pi[P_F] == 2:
        v = sigma[i]
        if live and np.isfinite(v) and j + 1 < len(ix):
            _place(fs, st, close, sd, v, j, i, 1, pf, pi)
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
def kernel(B, off, nbar, pusto, cl, dirs, sigma, div, pf, pi):  # xtask: длинная — день Run.step целиком, порядок важен
    """Run.step по всем торговым дням + постобработка exec_e.run. Возврат: (gross, cost, pos, free, ti, tf, st)."""
    n = len(nbar)
    ix = np.flatnonzero(~pusto)
    nix = len(ix)
    need = pi[P_CH] != 0 or pi[P_US] == 1 or pi[P_UT] == 1
    atr = _atr(B, off, nbar, pusto, cl) if need else np.empty(0)
    su = _unit(pi[P_US], atr, cl, pusto) if pi[P_US] else sigma
    tu = _unit(pi[P_UT], atr, cl, pusto) if pi[P_UT] else sigma
    fs, st = np.zeros(11), np.zeros(19, np.int64)
    ti, tf = np.full((nix + 1, 8), -1, np.int64), np.zeros((nix + 1, 2))
    acc, pos = np.zeros((3, n)), np.zeros(n)                # gross, cost, used по дням
    for j in range(nix):
        i = ix[j]
        nb = nbar[i]
        fs[DV] = div[i] * pf[F_NAL] if st[TR] else 0.0
        if st[TR]:
            st[I0] = 0
            fs[SH] += div[i]
        if st[OK] and st[ODUE] == j:
            _try_order(fs, st, ti, tf, acc, i, j, B, off, nbar, cl, dirs, sigma, su, tu, atr, ix, pf, pi)
        elif st[TR]:
            _scan(fs, st, ti, tf, acc, i, j, 0, B, off, nbar, cl, su, tu, atr, ix, pf, pi)
        close = cl[i]
        if st[XO]:
            if st[TR]:
                _exit(fs, st, ti, tf, acc, i, j, nb - 1, close, st[XR], pf, pi, ix)
            st[XO] = 0
        if st[TR]:
            lv = 0
            if pi[P_HTAU] and j - st[EJ] >= pi[P_TAU] and j > st[EJ]:
                lv = R_VREMYA
            elif pi[P_EXS] and dirs[i] != st[SD]:
                lv = R_SIGNAL
            if lv:
                sg = sigma[i]
                if pi[P_VLIM] and j + 1 < nix and np.isfinite(sg):
                    st[XN] += 1
                    st[XO], st[XR], fs[XX] = 1, lv, _limit_level(-st[SD], close, sg, pf, pi)
                else:
                    _exit(fs, st, ti, tf, acc, i, j, nb - 1, close, lv, pf, pi, ix)
        if st[TR]:
            acc[0, i] += st[SD] * math.log((close + fs[DV]) / fs[REF])
            fs[DV], fs[REF] = 0.0, close
            acc[2, i] += nb - st[I0]
            st[I0] = nb
        if st[BLK] and dirs[i] != st[BLK]:
            st[BLK] = 0
        sg, sd = sigma[i], dirs[i]
        if not (st[TR] or st[OK] or st[BLK] or not sd or not np.isfinite(sg)):
            if not pi[P_HK]:
                _enter(fs, st, ti, tf, acc, i, j, nb - 1, close, nb, sg, i, sd, nb, pf, pi, ix, su, tu)
            elif j + 1 < nix:
                _place(fs, st, close, sd, sg, j, i, 0, pf, pi)
        pos[i] = float(st[SD]) if st[TR] else 0.0
    if nix and st[TR]:
        acc[1, ix[-1]] += pf[F_COMM]
    gross, free = np.zeros(n), np.zeros(n)
    for j in range(1, nix):
        gross[ix[j - 1]] += acc[0, ix[j]]
        free[ix[j - 1]] = 1.0 - acc[2, ix[j]] / nbar[ix[j]]
    if nix:
        free[ix[-1]] = 1.0 - abs(pos[ix[-1]])
    for i in range(n):
        if pusto[i]:
            pos[i] = pos[i - 1] if i else 0.0
    return gross, acc[1].copy(), pos, free, ti[:st[NT]], tf[:st[NT]], st
