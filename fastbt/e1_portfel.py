"""Ж-051, итерация 3: топ-k по рангу |сила|/σ (проект docs/otchety/2026-10-28_zh051_iteracii.md §2.3).

Чистый модуль без данных. sig — позиции Sk (T, n) в {-1, 0, 1} (после фильтров входа), ball — баллы (T, n).
Решение о входе в день d — только по ball[d] (данные до 18:50 d), исполнение — d+1 (ось vhod, не здесь).
Слот занят днями сигнальной позиции сделки; сделка, закончившаяся в день e, освобождает слот с e+1
(без освобождения «на том же close»). Открытая позиция не вытесняется. Вес позиции не меняется (↺ ❓2).
k = "net"/None -> sig возвращается побайтно (golden).
"""

from __future__ import annotations

import numpy as np


def ball_sily(strength: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Балл s[t,d] = |strength[t,d]| / σ[t,d], обе — на close d (строка d), без сдвигов. σ<=0/NaN -> NaN."""
    st = np.abs(np.asarray(strength, float))
    sg = np.asarray(sigma, float)
    ok = np.isfinite(sg) & (sg > 0)
    return np.where(ok, st / np.where(ok, sg, 1.0), np.nan)


def _run_end(sig: np.ndarray) -> np.ndarray:
    """end[d, t] — последний день серии одного знака, в которой лежит день d (для pos != 0)."""
    n_d = sig.shape[0]
    end = np.arange(n_d)[:, None] * np.ones(sig.shape[1], int)
    for d in range(n_d - 2, -1, -1):
        same = (sig[d] != 0) & (sig[d + 1] == sig[d])
        end[d] = np.where(same, end[d + 1], d)
    return end


def raspredelit(sig, ball, k, sektor=None, n_sek=None) -> np.ndarray:
    """Оставляет сделки Sk, вошедшие в топ-k по баллу среди свободных слотов дня старта.

    Ничья — порядок тикеров (меньший индекс первым). NaN-балл — последним, но в пределах свободных слотов
    допускается. Не прошедшая сделка зануляется целиком. sektor (n,) + n_sek — лимит одновременных сделок сектора.
    """
    sig = np.asarray(sig, float)
    if k is None or k == "net":
        return sig.copy()
    ball = np.asarray(ball, float)
    if int(k) != k or k < 1:
        raise ValueError(f"k должен быть целым >= 1 или 'net', получено {k!r}")
    if ball.shape != sig.shape or sig.ndim != 2:
        raise ValueError("sig и ball — массивы (T, n) одной формы")
    if (sektor is None) != (n_sek is None):
        raise ValueError("sektor и n_sek задаются вместе")
    n_d, n_t = sig.shape
    sek = None if sektor is None else np.asarray(sektor)
    end = _run_end(sig)
    prev = np.vstack([np.zeros(n_t), sig[:-1]])
    acc_end = np.full(n_t, -1)
    out = np.zeros_like(sig)
    for d in range(n_d):
        cand = np.flatnonzero((sig[d] != 0) & (sig[d] != prev[d]))
        busy = acc_end >= d
        if len(cand):
            key = np.where(np.isnan(ball[d, cand]), -np.inf, ball[d, cand])
            for t in cand[np.argsort(-key, kind="stable")]:
                if busy.sum() >= k or (sek is not None and (busy & (sek == sek[t])).sum() >= n_sek):
                    continue
                acc_end[t] = end[d, t]
                busy[t] = True
                out[d : end[d, t] + 1, t] = sig[d, t]
    return out
