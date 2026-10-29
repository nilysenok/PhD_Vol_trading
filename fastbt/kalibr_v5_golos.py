"""П6 (Ж-127): голосование ансамбля по позициям автомата v2 (`exec_e.run` / `exec_e_v4sloj.run_v4`) для top-10 точек тройки.

Позиция члена — `Result.pos[t]` автомата (позиция на close t, как у A и у `execlayer`): вызывающий передаёт `poz_fn(kl, k) -> pos[T]`
по ключу точки (`kalibr_v5.klyuch`) и номеру тикера — прогон автомата по sig точки на ВСЕХ днях пула, не только в окне года t.
Голос — тот же `execlayer.vote` и та же доля `grids.VOTE_FRAC`, что в `kalibr_v5.ansambl`: ненулевых > доля·k, знак — по сумме.
Голос дня t берёт позиции дня t (не t+1): позиция на close t известна на close t.
"""
from __future__ import annotations

import numpy as np

from fastbt import grids
from fastbt.execlayer import vote


def golos_v2(poz_fn, chleny: list, k: int, n_dnej: int, s_t: int, e_t: int) -> np.ndarray | None:
    """Позиции ансамбля на [s_t; e_t) по позициям автомата членов; членов нет — None."""
    if not chleny:
        return None
    rows = [np.asarray(poz_fn(c["kl"], k), float) for c in chleny]
    if any(r.shape != (n_dnej,) for r in rows):
        raise ValueError("poz_fn: позиции длины пула (по всем дням, не только окно года)")
    out = np.zeros(n_dnej)
    vote(np.ascontiguousarray(np.stack(rows)), len(rows), s_t, e_t, out, grids.VOTE_FRAC)
    return out[s_t:e_t]
