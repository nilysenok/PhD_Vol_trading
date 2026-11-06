"""Волна D, поток LGBM (Ж-206): окна обучения, веса по возрасту, блоки обновления — чистые функции (правила RD1, RD2 в GATES_D_lgbm.md)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Tm:
    """Временная конфигурация: win — последние/ближайшие win пар (0 — расширяющееся), hl — полураспад весов в строках (0 — без весов),
    freq — d (ежедневно) | w (неделя ISO) | m (месяц) | adapt (месяц + смена режима)."""
    win: int = 0
    hl: float = 0.0
    freq: str = "d"


def nearest(idx, centre: float, win: int) -> np.ndarray:
    """win пар, ближайших по времени к centre (|t − centre|, ничья — раньше), по возрастанию t. win = 0 или win ≥ len — все пары.
    В прогнозе на дату d centre = d и пары лежат слева от d, поэтому это ровно «последние win пар»."""
    idx = np.asarray(idx)
    if not win or len(idx) <= win:
        return idx
    return np.sort(idx[np.argsort(np.abs(idx - centre), kind="stable")[:win]])


def weights_full(n_rows: int, centre: float, hl: float):
    """Веса по строкам 0…n−1: 0,5^(|t − centre|/hl) (в прогнозе centre = d: 0,5^((d − t)/hl) как `s4/fit.weights`); hl = 0 — без весов (None)."""
    if not hl:
        return None
    return 0.5 ** (np.abs(np.arange(n_rows) - centre) / float(hl))


def block_ids(dates, freq: str, flip=None) -> np.ndarray:
    """Номер блока обновления для каждой даты сетки; якорь блока — его первая дата. flip[i] — смена режима в строке i (для adapt)."""
    n = len(dates)
    if freq == "d":
        return np.arange(n)
    di = pd.DatetimeIndex(dates)
    if freq == "w":
        iso = di.isocalendar()
        key = (iso["year"].to_numpy(np.int64) * 100 + iso["week"].to_numpy(np.int64))
    elif freq in ("m", "adapt"):
        key = di.year.to_numpy(np.int64) * 100 + di.month.to_numpy(np.int64)
    else:
        raise ValueError(freq)
    new = np.r_[True, key[1:] != key[:-1]]
    if freq == "adapt":
        new = new | np.asarray(flip, bool)
    return np.cumsum(new) - 1


def blocks(grid, ids: np.ndarray) -> list:
    """[(строка якоря, [строки дат блока])]; строки F, не позиции сетки."""
    grid = np.asarray(grid, int)
    out = []
    for b in np.unique(ids):
        ds = grid[ids == b]
        out.append((int(ds[0]), [int(d) for d in ds]))
    return out
