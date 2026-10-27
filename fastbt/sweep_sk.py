"""Адаптер Sk для движка перебора: одна стратегия A (S1…S6) на всех тикерах, ряд до 18:50 (Н6), ≤ 2021.

Движок расчёта не переписан: сигналы — семейство `fastbt.signals` (`get_family(sid).generate`),
сделки — `batch.trades_one`, позиции — `stats.pos_from_trades`, счёт — `s9.Engine.port`
(комиссия с обеих сторон, плата за выход на последнем баре, ставка ЦБ — как в боевом портфеле).
Параметры сигнала — ключи `sigprov.SIG_GRIDS[sid]`; остальные — RM, обязаны быть узлами `grids.rm_grid(sid)`.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from fastbt import cache, grids, msp, s9
from fastbt.batch import trades_one
from fastbt.data import TICKERS, load_ticker, year_bounds
from fastbt.indicators import compute_base
from fastbt.signals.interface import get_family
from fastbt.sigprov import SIG_GRIDS
from fastbt.stats import pos_from_trades
from fastbt.unit import Unit

START_YEAR = 2018
END = pd.Timestamp("2022-01-01")


class _Mini(s9.Engine):
    """Портфель из позиций без кэша A: нужны только поля, которые читает `Engine.port`."""

    def __init__(self, dates, r_date, first, lr_row):
        self.dates = pd.DatetimeIndex(dates)
        self.r = dict(date=r_date)
        self.first, self.lr_row = first, lr_row
        self.cnt = np.bincount(r_date, minlength=len(self.dates)).astype(float)
        rate = msp.rate_cbr()
        kr = rate.reindex(rate.index.union(self.dates)).sort_index().ffill().bfill()
        self.kr = kr.reindex(self.dates).values
        self.rf = self.kr / 365.0
        self.cal = msp.cal_days(self.dates)


class SkEvaluator:
    def __init__(self, sid: str, tickers=None):
        self.sid = sid
        self.tickers = list(tickers or TICKERS)
        self.sig_names = set(SIG_GRIDS[sid])
        self._td = {}
        for t in self.tickers:
            td = load_ticker(t)
            assert pd.Timestamp(td.dates.max()) < END, "ряд содержит даты ≥ 2022"
            self._td[t] = (td, compute_base(td.close, td.high, td.low, td.open, td.volume))

    def series_digest(self) -> str:
        return hashlib.sha1("".join(self._td[t][0].digest() for t in self.tickers).encode()).hexdigest()

    def code_id(self) -> str:
        return cache.code_hash()

    def signals(self, sp: dict) -> dict:
        fam = get_family(self.sid)
        return {t: Unit(t, self.sid, td, ind, fam.generate(td, ind, [dict(sp)]))
                for t, (td, ind) in self._td.items()}

    def _rm_index(self, u: Unit, p: dict) -> int:
        rm = {k: v for k, v in p.items() if k not in self.sig_names}
        need = {k for k in u.rm[0] if not k.startswith("_")}
        if need - set(rm):
            raise KeyError(f"не заданы параметры RM: {sorted(need - set(rm))}")
        for i, d in enumerate(u.rm):
            if all(d[k] == rm[k] for k in need):
                return i
        raise KeyError(f"точка RM не узел сетки {self.sid}: {rm}")

    def pnl(self, sigs: dict, p: dict, comm: float, end: pd.Timestamp) -> pd.DataFrame:
        pos, lr, tick, bar, date = [], [], [], [], []
        for ti, (t, u) in enumerate(sigs.items()):
            i0 = year_bounds(u.dates, START_YEAR)[0]
            i1 = int(np.searchsorted(u.dates, np.datetime64(end)))
            te, tx, td = trades_one(u, 0, u.rmP[self._rm_index(u, p)], u.trend, end=i1)
            full = pos_from_trades(te, tx, td, len(te), u.n)
            b = np.arange(i0, i1)
            pos.append(full[b])
            lr.append(u.lr[b])
            date.append(u.dates[b])
            tick.append(np.full(len(b), ti))
        dts, inv = np.unique(np.concatenate(date), return_inverse=True)
        pos, lr, tick = np.concatenate(pos), np.concatenate(lr), np.concatenate(tick)
        first = np.r_[True, tick[1:] != tick[:-1]]
        eng = _Mini(dts, inv, first, lr)
        df = eng.port(pos, comm=comm)[["excess", "exposure", "turnover"]]
        dp = np.diff(pos, prepend=0.0)
        dp[first] = pos[first]
        last = np.r_[first[1:], True]
        sides = (np.abs(dp) > 0) + (last & (pos != 0))
        df["sides"] = np.bincount(inv, sides.astype(float), len(dts)) / eng.cnt
        return df
