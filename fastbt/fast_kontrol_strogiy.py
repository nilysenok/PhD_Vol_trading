"""Контроль к2 при строгом исполнении (лист CE-3; Ж-086, Ж-088 п. 2). Ядро сдвига с двумя ногами (n, dn).

К2 сдвигает только затвор g внутри своей пары (g'[j] = g[(j − lag[тикер]) mod T]); позиция = A·g'. Ноги рынка (n, dn)
не сдвигаются — это данные дня. Исполнение — `strogoe.port_strogiy` (вход и выход по open 10:00 следующего дня, плата за шорт).
Лаги — те же, что у старого к2: `fast_kontrol.draw_lags` (seed 20261027, ≥ 21 день); p — `fast_kontrol.p_perm`.
`fast_kontrol.py` и `s9.py` не правятся. Лаг 0 = исходная точка побайтно (mutations.md № 9).
"""
from __future__ import annotations

import numpy as np

from fastbt import strogoe
from fastbt.fast_kontrol import MIN_LAG, N_SHIFT, SEED, draw_lags, p_perm  # noqa: F401  (p_perm — для вызывающих)


def _geometry(eng) -> tuple:
    """(начало пары, длина пары, номер строки в паре) по строкам движка."""
    first = np.asarray(eng.first, bool)
    starts = np.flatnonzero(first)
    lens = np.diff(np.r_[starts, len(first)])
    pstart = np.repeat(starts, lens)
    return pstart, np.repeat(lens, lens), np.arange(len(first)) - pstart


def shift_g(eng, g, lag) -> np.ndarray:
    """g'[j] = g[pstart + (loc − lag[тикер]) mod plen]; lag — число (на все тикеры) или вектор по тикерам."""
    pstart, plen, loc = _geometry(eng)
    lag = np.asarray(lag, np.int64)
    lg = lag if lag.ndim == 0 else lag[np.asarray(eng.r["tick"], np.int64)]
    return np.asarray(g, float)[pstart + (loc - lg) % plen]


def shift_strogiy(eng, A, g, lag, comm, nogi, zaem: float = 0.05):
    """Портфель позиции A·g'(lag) на строгом исполнении (колонки `port_strogiy`)."""
    pos = np.asarray(A, float) * shift_g(eng, g, lag)
    return strogoe.port_strogiy(eng, pos, comm, zaem, nogi)


def sharpe(excess) -> float:
    """Шарп избыточной доходности: ddof=1, √252 (одно определение с `fast_kontrol.shift_sharpe`)."""
    x = np.asarray(excess, float)
    sd = x.std(ddof=1)
    return float(x.mean() * np.sqrt(252.0) / sd) if sd > 1e-12 else 0.0


def lags_k2(eng, n_shift: int = N_SHIFT, seed: int = SEED, min_lag: int = MIN_LAG) -> np.ndarray:
    """LAG[b, тикер] — те же, что в старом к2 (длина ряда тикера = число его строк в самой короткой паре)."""
    first = np.asarray(eng.first, bool)
    starts = np.flatnonzero(first)
    lens = np.diff(np.r_[starts, len(first)])
    tick = np.asarray(eng.r["tick"], np.int64)
    t_tick = np.full(int(tick.max()) + 1, np.iinfo(np.int64).max)
    np.minimum.at(t_tick, tick[starts], lens)
    t_tick[t_tick == np.iinfo(np.int64).max] = 0
    return draw_lags(t_tick, n_shift, seed, min_lag)


def k2_sharpes_strogiy(eng, A, g, comm, nogi, LAG=None, zaem: float = 0.05) -> np.ndarray:
    """Шарпы позиций A·g'_b при строгом исполнении, b по строкам LAG (нет — `lags_k2`)."""
    LAG = lags_k2(eng) if LAG is None else LAG
    return np.array([sharpe(shift_strogiy(eng, A, g, LAG[b], comm, nogi, zaem)["excess"].values)
                     for b in range(LAG.shape[0])])
