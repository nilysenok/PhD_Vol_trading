"""E1 (X1 = D1): доверие к бустингу по состоянию. F = M^(1−π)·C^π, π = π^(s_d), s_d — экстремальный M10-бар дня d; π^(s) — fixed share (формула NTP2 V4, `models_v2/s11/v4_ntp_state.fixed_share`): логит веса C,
шаг −η·(QLIKE(C) − QLIKE(M)) по ЗАКРЫТОЙ паре (t⁺ ≤ d) СВОЕГО состояния, затем подмешивание α/2; η_d = 1/медиана QLIKE обеих моделей по всем закрытым парам (как NTP2); старт π = ½.
α^(s) ∈ GRID — помесячно по последним VAL закрытым парам состояния (контрфактический π этих пар — по данным, закрытым к их дню). Чистые функции от массивов: файлов нет."""
from __future__ import annotations

import bisect

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from models_v2.s11.v4_ntp_state import combine, qlike
from models_v2.s15 import de0_groups as G

GRID, VAL, A_DEF, EPS, Q_EXTREME = (0.01, 0.02, 0.05), 126, 0.02, 1e-12, 0.99


def closed_date(t_plus, date):
    """День, с которого пара доступна: t⁺ (мутация «закрыто + 1» подменяет на дату прогноза пары)."""
    return t_plus


def past_end(p: int) -> int:
    """Бары, учтённые в пороге дня на позиции p склада: СТРОГО до p (мутация: p + 1 — день d входит в порог)."""
    return p


def cutoff(dates, i):
    """Граница данных выбора α в день i: дата дня i (мутация: последняя дата выборки — «по 2022+»)."""
    return dates[i]


def states(store: dict, dates) -> np.ndarray:
    """s_d = 1, если max|r| M10-баров дня d > 99-й перцентиль |r| баров сессий < d (определение DE0 `de0_groups.extreme_flag` без отсечки MIN_PAST: порог есть всегда, раз в складе есть прошлое).
    День без баров или без прошлых баров — 0 (норма); число таких дней возвращает вызывающий через `np.isfinite(maxabs)`."""
    st, absr = G.day_stats(store, dates), np.abs(store["r"])
    out = np.zeros(len(dates), int)
    for i in range(len(dates)):
        past = absr[: past_end(st["pos"][i])]
        past = past[np.isfinite(past)]
        if len(past) and np.isfinite(st["maxabs"][i]):
            out[i] = int(st["maxabs"][i] > np.quantile(past, Q_EXTREME))
    return out


def _pass(qc, qm, s, dates, tp, alpha_at) -> tuple:
    """Один онлайн-проход. -> (π по состояниям (n, 2) после учёта пар, закрытых к дню i; η по дням). alpha_at(i) -> (α₀, α₁) на день i (до учёта новых пар дня)."""
    n = len(qc)
    avail = np.asarray(closed_date(np.asarray(tp), np.asarray(dates)))
    order, lg, pool, ptr, eta = np.argsort(avail, kind="stable"), [0.0, 0.0], [], 0, np.nan
    pis, etas = np.empty((n, 2)), np.full(n, np.nan)
    for i in range(n):
        new = []
        while ptr < n and avail[order[ptr]] <= dates[i]:
            new.append(order[ptr])
            ptr += 1
        for j in new:
            bisect.insort(pool, qc[j])
            bisect.insort(pool, qm[j])
        if pool:
            eta = 1.0 / max(0.5 * (pool[len(pool) // 2] + pool[(len(pool) - 1) // 2]), EPS)
        a = alpha_at(i)
        for j in new:
            k = int(s[j])
            lg[k] -= (eta if np.isfinite(eta) else 0.0) * (qc[j] - qm[j])
            lg[k] = float(logit((1.0 - a[k]) * expit(lg[k]) + a[k] / 2.0))
        pis[i], etas[i] = expit(lg), eta
    return pis, etas


def select_alpha(loss: dict, s, avail, dates, i, k) -> float:
    """α состояния k на день i: минимум средней потери по последним VAL ЗАКРЫТЫМ парам состояния k (avail ≤ cutoff). loss[α][j] — QLIKE ряда F пары j при её контрфактическом π. Нет пар — A_DEF; ничья — первая в GRID."""
    js = [j for j in np.argsort(avail, kind="stable") if s[j] == k and avail[j] <= cutoff(dates, i)][-VAL:]
    if not js:
        return A_DEF
    return GRID[int(np.argmin([float(np.mean(loss[a][js])) for a in GRID]))]


def run(rv, M, C, s, dates, tp) -> dict:
    """Все дни: -> dict(pi, F, a0, a1, pis (n, 2), eta). Для каждого α из GRID — полный проход с одним α (контрфактический π пары j по её состоянию → потеря); затем боевой проход с помесячным выбором α по данным ≤ d."""
    rv, M, C, s, tp = (np.asarray(a) for a in (rv, M, C, s, tp))
    dates = np.asarray(dates)
    qc, qm, n = qlike(rv, C), qlike(rv, M), len(rv)
    avail = np.asarray(closed_date(tp, dates))
    loss = {}
    for a in GRID:
        pis, _ = _pass(qc, qm, s, dates, tp, lambda i, a=a: (a, a))
        loss[a] = qlike(rv, combine(M, C, pis[np.arange(n), s.astype(int)]))
    cur, alphas = [A_DEF, A_DEF], np.empty((n, 2))
    month = [None]

    def alpha_at(i):
        t = pd.Timestamp(dates[i])
        m = (t.year, t.month)
        if m != month[0]:
            month[0] = m
            cur[:] = [select_alpha(loss, s, avail, dates, i, k) for k in (0, 1)]
        alphas[i] = cur
        return tuple(cur)

    pis, eta = _pass(qc, qm, s, dates, tp, alpha_at)
    pi = pis[np.arange(n), s.astype(int)]
    return dict(pi=pi, F=combine(M, C, pi), a0=alphas[:, 0], a1=alphas[:, 1], pis=pis, eta=eta)
