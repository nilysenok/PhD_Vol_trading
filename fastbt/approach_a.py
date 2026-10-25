"""Подход A (V4:471-638; S5/S6 — V41:180-362): грубая сетка → соседи → слой исполнения →
топ-10 → голосование. Полная сетка (сигнал × RM) считается ОДИН раз на единицу
(позиции причинны: прогон до train_end = префикс полного прогона), окна лет — из сумм
сегментов. Порядок списков и устойчивые сортировки повторяют старый код (ничьи).
"""
from __future__ import annotations

import numpy as np

from fastbt import grids
from fastbt.batch import run_grid, trades_one
from fastbt.data import year_bounds
from fastbt.execlayer import exec_layer, vote
from fastbt.signals.interface import WARMUP
from fastbt.stats import NF, pos_from_trades, seg_stats_pos, window


def _desc(items, key):
    return sorted(items, key=key, reverse=True)


class ApproachA:
    def __init__(self, u, cfg):
        self.u, self.cfg = u, cfg
        self.v41 = u.sid in grids.V41_SIDS
        self.exec_grid = grids.exec_grid()
        K, R = len(u.sigs), len(u.rm)
        sp = np.repeat(np.arange(K), R)
        rm = np.tile(np.arange(R), K)
        self.full = run_grid(u, sp, rm, u.rmP, u.trend, cfg.comm).reshape(K, R, len(u.nbars), NF)
        self._raw = {}
        self._ex = {}

    def raw_pos(self, i, r):
        k = (i, r)
        if k not in self._raw:
            te, tx, td = trades_one(self.u, i, self.u.rmP[r], self.u.trend)
            self._raw[k] = pos_from_trades(te, tx, td, len(te), self.u.n)
        return self._raw[k]

    def exec_pos(self, i, r, e):
        ep = self.exec_grid[e]
        return exec_layer(self.raw_pos(i, r), self.u.strength[i], ep["consec_entry"],
                          ep["min_hold"], ep["cooldown_bars"], ep["min_strength"], self.u.n)

    def exec_stats(self, i, r, e):
        k = (i, r, e)
        if k not in self._ex:
            out = np.zeros((len(self.u.nbars), NF))
            seg_stats_pos(self.exec_pos(i, r, e), self.u.lr, self.u.segb, self.u.seg_of,
                          WARMUP, self.u.n, self.cfg.comm, out)
            self._ex[k] = out
        return self._ex[k]

    def year(self, Y):
        """Отбор года Y: список топ-10 (i_sp, i_rm, i_exec) или None."""
        u = self.u
        train_end = year_bounds(u.dates, Y)[self.cfg.bound()]
        if train_end <= WARMUP or train_end >= u.n:
            return None
        j = u.seg_index(train_end)
        sh, mdd, ex, tpy = window(self.full, u.nbars, j)
        R = len(u.rm)
        coarse = grids.coarse_idx(u.sid, R)
        res = [(i, r) for i in range(len(u.sigs)) for r in coarse]
        ok = (sh > 0) & (mdd > -0.30) & (ex > 5)
        if self.v41:
            ok &= tpy >= 1.0
        passing = [c for c in res if ok[c]]
        if not passing:
            passing = _desc(res, lambda c: sh[c])[:10]
        top50 = _desc(passing, lambda c: sh[c])[:50]
        dedup = dict.fromkeys(top50)
        for i, r in top50:
            for dlt in (-2, -1, 1, 2):
                q = r + dlt
                if 0 <= q < R and (i, q) not in dedup:
                    dedup[(i, q)] = None
        top20 = _desc(list(dedup), lambda c: sh[c])[:20]
        combos = []
        for i, r in top20:
            for e in range(len(self.exec_grid)):
                s, _, _, _ = window(self.exec_stats(i, r, e), u.nbars, j)
                combos.append(((i, r, e), float(s)))
        top10 = _desc(combos, lambda c: c[1])[:10]
        return [c[0] for c in top10] if top10 else None

    def ensemble(self, top10):
        u = self.u
        stack = np.stack([self.exec_pos(i, r, e) for i, r, e in top10])
        out = np.zeros(u.n)
        vote(stack, len(top10), 0, u.n, out)
        return out

    def run(self, years=grids.A_TEST_YEARS):
        """{Y: (top10, ансамбль)}."""
        out = {}
        for Y in years:
            top = self.year(Y)
            if top is not None:
                out[Y] = (top, self.ensemble(top))
        return out
