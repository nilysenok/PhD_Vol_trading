"""Фильтр оборота (решение владельца 25.10, 19:30): минимальное удержание позиции и запрет
повторного входа в ту же пару в течение N баров после выхода; (удержание, N) — по Шарпу на
окне отбора подхода (A — окно обучения, B/C/D — валидация). Применяется к итоговой позиции.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from fastbt import grids
from fastbt.data import year_bounds
from fastbt.signals.interface import WARMUP
from fastbt.stats import sharpe_window


@njit(cache=True)
def tfilter(raw, min_hold, nban):
    """Знак позиции — с удержанием ≥ min_hold баров и запретом входа nban баров после выхода
    (nban = 0 — без запрета, разворот разрешён); размер — |raw| при совпадении знака,
    иначе последний размер."""
    n = len(raw)
    out = np.zeros(n)
    cur = 0.0
    mag = 0.0
    held = 0
    last_exit = -1000000000
    for t in range(n):
        r = raw[t]
        sr = 1.0 if r > 0 else (-1.0 if r < 0 else 0.0)
        if cur != 0.0:
            held += 1
            if sr == cur:
                mag = abs(r)
            elif held >= min_hold:
                cur = 0.0
                last_exit = t
                if sr != 0.0 and nban == 0:
                    cur, mag, held = sr, abs(r), 0
        elif sr != 0.0 and (nban == 0 or t - last_exit > nban):
            cur, mag, held = sr, abs(r), 0
        out[t] = cur * mag
    return out


def apply(res: dict, u, cfg, kind: str):
    """res {Y: (params, pos, ...)} → тот же формат с отфильтрованной позицией; выбор в params."""
    b = cfg.bound()
    out = {}
    for Y, v in res.items():
        pos = v[1]
        if kind == "A":
            s, e = WARMUP, year_bounds(u.dates, Y)[b]
        else:
            s, e = year_bounds(u.dates, cfg.val_start)[b], year_bounds(u.dates, Y)[b]
        best, bp, bpos = -np.inf, None, pos
        for h in grids.TF_HOLD:
            for nb in grids.TF_REENTRY:
                f = tfilter(pos.astype(np.float64), int(h), int(nb))
                sh = sharpe_window(f, u.lr, s, e, cfg.comm) if e > s else -999.0
                if sh > best:
                    best, bp, bpos = sh, dict(min_hold=int(h), reentry_ban=int(nb)), f
        p0 = v[0]
        params = dict(p0, tf=bp) if isinstance(p0, dict) else {"base": p0, "tf": bp}
        out[Y] = (params, bpos) + tuple(v[2:])
    return out
