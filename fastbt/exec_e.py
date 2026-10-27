"""Семья E: внутридневное исполнение по правилам §2 (docs/zadaniya/2026-10-27_semya_E.md).

Один тикер, длинная позиция размером 1 капитал. Дни — `Day` (5-минутные бары o/h/l/c, флаг pusto).
sig[d] (1 — держать/входить) и sigma[d] (σ̂₁ на закрытии d) известны на закрытии d; заявка на d+1
исполняется только по барам d+1 и позже. Дни pusto выбрасываются из ленты тикера (как test_08_halts):
заявки не исполняются, позиция переносится, доходность паузы ложится целиком на следующий торговый
день, дни паузы не считаются в τ.

Учёт: логарифмы цен сделок. gross[t] — движение, случившееся в день t (от цены входа или close t-1
до цены выхода или close t); cost[t] — комиссия сторон, исполненных в день t; pnl = gross - cost.
Вход по close d (k=None) без уровней и времени даёт ровно A: gross[t+1] = pos[t]·lr[t], cost[t] = |Δpos|·c.
free[t] — доля дня t вне рынка (по барам): капитал, освободившийся внутри дня, получает ставку за остаток.
Правило «одна сделка на эпизод сигнала»: после stop/take/vremya и после неисполненной заявки новый вход
не ставится, пока sig не станет 0. Тейк в баре входа не проверяется (консервативно), стоп — проверяется.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Day:
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    pusto: bool = False


@dataclass(frozen=True)
class Params:
    k: float | None = None        # None — вход по close d; иначе лимит L = P_d·(1 − k·σ̂₁)
    f: str = "propusk"            # propusk | zakrytie_d1 | povtor
    a: float | None = None        # тейк T = P_вх·(1 + a·σ̂)
    b: float | None = None        # стоп S = P_вх·(1 − b·σ̂)
    tau: int | None = None        # временной стоп, торговых дней
    s: str = "fiks"               # fiks | ezhedn
    comm: float = 0.0004          # за сторону, доля
    exit_signal: bool = True      # выход по close первого дня с sig == 0


@dataclass
class Trade:
    ticker: str
    d_sig: int
    entry_day: int
    entry_bar: int
    entry_px: float
    exit_day: int | None = None
    exit_bar: int | None = None
    exit_px: float | None = None
    reason: str | None = None     # stop | take | vremya | signal; None — открыта на конце
    fee_in: float = 0.0
    fee_out: float = 0.0


@dataclass
class Result:
    trades: list
    gross: np.ndarray
    cost: np.ndarray
    pos: np.ndarray
    free: np.ndarray
    fdr: np.ndarray

    @property
    def pnl(self):
        return self.gross - self.cost


def run(ticker, days, sig, sigma, p: Params, rf_cal=None) -> Result:
    """rf_cal[t] — ставка·календарные дни интервала, на который приходится день t (Н2: [d, d_next))."""
    r = _Run(ticker, days, np.asarray(sig), np.asarray(sigma, float), p)
    for j, i in enumerate(r.ix):
        r.step(j, i)
    for i, d in enumerate(days):
        if d.pusto:
            r.pos[i] = r.pos[i - 1] if i else 0.0
            r.free[i] = 1.0 - r.pos[i]
    rf = np.zeros(len(days)) if rf_cal is None else np.asarray(rf_cal, float)
    return Result(r.trades, r.gross, r.cost, r.pos, r.free, r.free * rf)


class _Run:
    def __init__(s, ticker, days, sig, sigma, p):
        n = len(days)
        s.t, s.days, s.sig, s.sigma, s.p = ticker, days, sig, sigma, p
        s.ix = [i for i, d in enumerate(days) if not d.pusto]
        s.gross, s.cost, s.pos = np.zeros(n), np.zeros(n), np.zeros(n)
        s.used, s.free = np.zeros(n), np.ones(n)
        s.trades, s.tr, s.order = [], None, None
        s.ref = s.i0 = s.sg = s.ej = None
        s.blocked = False

    # --- вход / выход ---
    def _enter(s, i, j, b, px, i0, sg, d_sig):
        s.tr = Trade(s.t, d_sig, i, b, px, fee_in=s.p.comm)
        s.trades.append(s.tr)
        s.cost[i] += s.p.comm
        s.ref, s.i0, s.sg, s.ej = px, i0, sg, j

    def _exit(s, i, b, px, reason):
        tr = s.tr
        tr.exit_day, tr.exit_bar, tr.exit_px, tr.reason, tr.fee_out = i, b, px, reason, s.p.comm
        s.gross[i] += math.log(px / s.ref)
        s.cost[i] += s.p.comm
        s.used[i] += b + 1 - s.i0
        s.tr = s.ref = s.i0 = None
        if reason != "signal":
            s.blocked = True

    def _levels(s, j):
        sg = s.sg
        if s.p.s == "ezhedn" and j > s.ej:
            v = s.sigma[s.ix[j - 1]]
            sg = v if np.isfinite(v) else sg
        if not (np.isfinite(sg) and sg > 0):
            return None, None
        px = s.tr.entry_px
        return (px * (1 - s.p.b * sg) if s.p.b is not None else None,
                px * (1 + s.p.a * sg) if s.p.a is not None else None)

    def _scan(s, i, j, d, b0):
        """Бары b0.. дня: стоп раньше тейка; гэп — по open."""
        S, T = s._levels(j)
        for b in range(b0, len(d.c)):
            if S is not None and d.l[b] < S:
                return s._exit(i, b, d.o[b] if d.o[b] < S else S, "stop")
            if T is not None and d.h[b] > T:
                return s._exit(i, b, d.o[b] if d.o[b] > T else T, "take")

    # --- день ---
    def step(s, j, i):
        d, nb = s.days[i], len(s.days[i].c)
        if s.tr is not None:
            s.i0 = 0
        if s.order is not None and s.order["due"] == j:
            s._try_order(i, j, d)
        elif s.tr is not None:
            s._scan(i, j, d, 0)
        close = d.c[-1]
        if s.tr is not None and j > s.ej:
            if s.p.tau is not None and j - s.ej >= s.p.tau:
                s._exit(i, nb - 1, close, "vremya")
            elif s.p.exit_signal and not s.sig[i]:
                s._exit(i, nb - 1, close, "signal")
        if s.tr is not None:
            s.gross[i] += math.log(close / s.ref)
            s.ref = close
            s.used[i] += nb - s.i0
            s.i0 = nb
        if not s.sig[i]:
            s.blocked = False
        s._decide(i, j, close, nb)
        s.pos[i] = 1.0 if s.tr is not None else 0.0
        s.free[i] = 1.0 - s.used[i] / nb

    def _decide(s, i, j, close, nb):
        sg = s.sigma[i]
        if s.tr is not None or s.order is not None or s.blocked or not s.sig[i] or not np.isfinite(sg):
            return
        if s.p.k is None:
            s._enter(i, j, nb - 1, close, nb, sg, i)
        elif j + 1 < len(s.ix):
            s.order = dict(L=close * (1 - s.p.k * sg), due=j + 1, retry=False, d_sig=i)

    def _try_order(s, i, j, d):
        o, s.order = s.order, None
        hit = np.flatnonzero(d.l < o["L"])
        if hit.size:
            b = int(hit[0])
            px = d.o[b] if d.o[b] < o["L"] else o["L"]
            s._enter(i, j, b, px, b, s.sigma[o["d_sig"]] if not o["retry"] else o["sg"], o["d_sig"])
            S, _ = s._levels(j)
            if S is not None and d.l[b] < S:
                s._exit(i, b, S, "stop")
            else:
                s._scan(i, j, d, b + 1)
            return
        close, nb = d.c[-1], len(d.c)
        if o["retry"] or s.p.f == "propusk":
            s.blocked = True
        elif s.p.f == "zakrytie_d1":
            s._enter(i, j, nb - 1, close, nb, s.sigma[i], o["d_sig"])
        elif s.p.f == "povtor":
            v = s.sigma[i]
            if np.isfinite(v) and j + 1 < len(s.ix):
                s.order = dict(L=close * (1 - s.p.k * v), due=j + 1, retry=True, d_sig=i, sg=v)
            else:
                s.blocked = True
