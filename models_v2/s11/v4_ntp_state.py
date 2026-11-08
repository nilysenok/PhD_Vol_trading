"""V4-NTP: причинное «доверие гибриду» π по прошлым парам (Ж-225 п. 4). Два прогноза: M = √(K2·K3) (чистый ML) и C8 (гибрид); D′ = QLIKE(C8) − QLIKE(M) на закрытой паре (плюс — ML лучше).
Пара (прогноз d, цель t⁺) доступна прогнозу дня D, если closed_date(пара) ≤ D (боевое правило: t⁺ ≤ D; тест подменяет на дату прогноза — «закрыто + 1»). Состояние бумаги — только по её закрытым парам;
панельное — по бумагам panel_mask (17; в LOTO 16). До MIN_PAIRS закрытых пар бумаги π = 0,5 (приор, как в B/C/E). Гиперпараметры фиксированы объявлением (GATES_V4.md, «Лист NTP»)."""
from __future__ import annotations

import bisect

import numpy as np
from scipy.special import expit, logit

HL, KAPPA, MIN_PAIRS, ALPHA, STEP, EPS = 60.0, 250.0, 60, 0.02, 0.02, 1e-12


def closed_date(t_plus, date):
    """День, с которого пара доступна: дата цели t⁺ (мутация «закрыто + 1» подменяет на дату прогноза)."""
    return t_plus


def qlike(rv, f):
    r = np.asarray(rv, float) / np.asarray(f, float)
    return r - np.log(r) - 1.0


def combine(M, C, pi):
    """F = M^(1−π)·C8^π; на концах точно M (π = 0) и C8 (π = 1) — побитно (exp(ln x) ≠ x)."""
    M, C, pi = (np.asarray(a, float) for a in (M, C, pi))
    mid = np.exp((1.0 - pi) * np.log(M) + pi * np.log(C))
    return np.where(pi >= 1.0, C, np.where(pi <= 0.0, M, mid))


def _days(date):
    days, starts = np.unique(date, return_index=True)
    return days, starts, list(starts[1:]) + [len(date)]


def ewma_state(rv, M, C, tix, date, tp, nT, pm, hl=None, kappa=None, min_pairs=None) -> tuple:
    """Строки отсортированы по (date, ticker). -> (m, s, pan, cnt) по строкам: m — EWMA D′ бумаги (полураспад hl пар), стянутое к панельному
    (n·m_i + κ·m_панель)/(n + κ), n = (Σω)²/Σω²; s — EWMA с. о. D′ бумаги; pan — панельное m дня (EWMA по панели, затухание на каждый торговый день); cnt — закрытые пары бумаги.
    m, s = NaN, пока у бумаги меньше min_pairs закрытых пар."""
    hl, kappa, min_pairs = HL if hl is None else hl, KAPPA if kappa is None else kappa, MIN_PAIRS if min_pairs is None else min_pairs
    n, lo = len(rv), 0.5 ** (1.0 / hl)
    x = qlike(rv, C) - qlike(rv, M)
    avail = np.asarray(closed_date(np.asarray(tp), np.asarray(date)))
    order = np.argsort(avail, kind="stable")
    num, num2, den, den2, cnt = (np.zeros(nT) for _ in range(5))
    pn = pn2 = pd_ = 0.0
    m, s, pan, cr = (np.full(n, np.nan) for _ in range(4))
    ptr = 0
    for d, a, e in zip(*_days(date)):
        pn, pn2, pd_ = lo * pn, lo * pn2, lo * pd_
        while ptr < n and avail[order[ptr]] <= d:
            j = order[ptr]
            ptr += 1
            i = tix[j]
            num[i], num2[i], den[i], den2[i], cnt[i] = lo * num[i] + x[j], lo * num2[i] + x[j] ** 2, lo * den[i] + 1.0, lo * lo * den2[i] + 1.0, cnt[i] + 1
            if pm[i]:
                pn, pn2, pd_ = pn + x[j], pn2 + x[j] ** 2, pd_ + 1.0
        mp = pn / pd_ if pd_ > 0 else 0.0
        for r in range(a, e):
            i = tix[r]
            pan[r], cr[r] = mp, cnt[i]
            if cnt[i] < min_pairs:
                continue
            mi = num[i] / den[i]
            ne = den[i] ** 2 / den2[i]
            m[r] = (ne * mi + kappa * mp) / (ne + kappa)
            s[r] = np.sqrt(max(num2[i] / den[i] - mi * mi, EPS))
    return m, s, pan, cr


def pi_discount(m, s):
    """NTP1: π = σ(−m/s); нет оценки (NaN) — приор 0,5."""
    z = np.where(np.isfinite(m) & np.isfinite(s), m / np.where(s > 0, s, 1.0), np.nan)
    return np.where(np.isfinite(z), expit(-np.nan_to_num(z)), 0.5), z


def smooth(pi_star, tix, nT, step=None):
    """NTP4: π_t = π_{t−1} + clip(π*_t − π_{t−1}, ±step) по строкам бумаги; старт 0,5."""
    step = STEP if step is None else step
    out, prev = np.empty(len(pi_star)), np.full(nT, 0.5)
    for r in range(len(pi_star)):
        i = tix[r]
        prev[i] += np.clip(pi_star[r] - prev[i], -step, step)
        out[r] = prev[i]
    return out


def max_step(pi, tix, nT) -> float:
    """Наибольший |π_t − π_{t−1}| по бумагам (старт 0,5): проверка ограничения NTP4."""
    prev, mx = np.full(nT, 0.5), 0.0
    for p, i in zip(pi, tix):
        mx, prev[i] = max(mx, abs(p - prev[i])), p
    return float(mx)


def fixed_share(rv, M, C, tix, date, tp, nT, pm, alpha=None, min_pairs=None) -> tuple:
    """NTP2: веса {M, C8} ∝ exp(−η·накопленная QLIKE) по закрытым парам бумаги с подмешиванием α к равным весам на каждом шаге; η_d = 1/медиана QLIKE (обеих моделей) закрытых пар панели на день d.
    Состояние — логит веса C8 (при α = 0 без подмешивания нет ограничения снизу). -> (π, η) по строкам; π = 0,5, пока у бумаги < min_pairs закрытых пар."""
    alpha, min_pairs = ALPHA if alpha is None else alpha, MIN_PAIRS if min_pairs is None else min_pairs
    n = len(rv)
    qc, qm = qlike(rv, C), qlike(rv, M)
    avail = np.asarray(closed_date(np.asarray(tp), np.asarray(date)))
    order = np.argsort(avail, kind="stable")
    lg, cnt, pool = np.zeros(nT), np.zeros(nT), []
    pi, eta_r = np.full(n, 0.5), np.full(n, np.nan)
    ptr, eta = 0, np.nan
    for d, a, e in zip(*_days(date)):
        new = []
        while ptr < n and avail[order[ptr]] <= d:
            new.append(order[ptr])
            ptr += 1
        for j in new:
            if pm[tix[j]]:
                bisect.insort(pool, qc[j])
                bisect.insort(pool, qm[j])
        if pool:
            eta = 1.0 / max(0.5 * (pool[len(pool) // 2] + pool[(len(pool) - 1) // 2]), EPS)
        for j in new:
            i = tix[j]
            lg[i] -= (eta if np.isfinite(eta) else 0.0) * (qc[j] - qm[j])
            cnt[i] += 1
            if alpha > 0:
                lg[i] = logit((1.0 - alpha) * expit(lg[i]) + alpha / 2.0)
        for r in range(a, e):
            eta_r[r] = eta
            if cnt[tix[r]] >= min_pairs:
                pi[r] = expit(lg[tix[r]])
    return pi, eta_r
