"""Семья E: внутридневное исполнение по правилам §2 (docs/zadaniya/2026-10-27_semya_E.md).

Один тикер, длина 1 капитал. sig[d], sigma[d] (σ̂₁) известны на close d; заявка на d+1 исполняется
по барам d+1 и позже. Дни pusto выпадают из ленты (test_08_halts): позиция переносится, доходность
паузы — на интервал до неё, в τ не считаются. Плохие бары (пусто/NaN/≤0) — ValueError.

Учёт — соглашение A (msp.portfolio, Н2), логарифмы цен сделок. Строка t = интервал [close t, close
следующего торгового дня): pos[t] — позиция на close t; gross[t] = pos[t]·lr[t] у A; cost[t] — комиссии
исполнений по close t, внутридневные (лимит/стоп/тейк) — в строке начала их интервала; pnl = gross − cost.
Ставка: free[t] = 1 − доля баров следующего дня в рынке (у A 1 − pos[t]): позиция, открытая по close d,
ставку за день d не получает; после выхода по close x ставка идёт со строки x. fdr = free·rf_cal,
rf_cal[t] = rf/365·календарные дни [t, след. торговый день) (msp.cal_days); на днях pusto нули.
Избыточная = pnl + fdr − rf_cal.
Сигнал: sig == 0 на close любого дня удержания, включая день лимитного входа → выход по close
(позиция не доживает до d+2); zakrytie_d1 и повтор — только при sig[d+1] == 1. Повторный вход как A:
после stop/take/vremya/signal при sig == 1 — снова на том же close (cooldown A — параметр RM, тут 0).
↺ A не задаёт неисполненную заявку: вход блокируется до sig == 0. Тейк в баре входа не проверяется.
"""
from __future__ import annotations

import math

import numpy as np

from fastbt.exec_e_types import Day, Params, Result, Trade  # noqa: F401


def run(ticker, days, sig, sigma, p: Params, rf_cal=None) -> Result:
    """rf_cal[t] — ставка/365·календарные дни [t, следующий торговый день) (msp.cal_days, Н2)."""
    sig, sigma = np.asarray(sig), np.asarray(sigma, float)
    if not (len(days) == len(sig) == len(sigma)):
        raise ValueError("days, sig, sigma разной длины")
    for i, d in enumerate(days):
        if not d.pusto:
            a = np.stack([d.o, d.h, d.l, d.c]) if len(d.c) else None
            if a is None or not np.isfinite(a).all() or (a <= 0).any():
                raise ValueError(f"день {i}: пустой или нечисловой набор баров (нужен pusto=True)")
    r = _Run(ticker, days, sig, sigma, p)
    for j, i in enumerate(r.ix):
        r.step(j, i)
    n = len(days)
    gross, free = np.zeros(n), np.zeros(n)
    for j, i in enumerate(r.ix):                   # день i → интервал, начатый предыдущим торговым днём
        if j:
            gross[r.ix[j - 1]] += r.gross[i]
            free[r.ix[j - 1]] = 1.0 - r.used[i] / len(days[i].c)
    if r.ix:
        free[r.ix[-1]] = 1.0 - r.pos[r.ix[-1]]
    for i, d in enumerate(days):
        if d.pusto:
            r.pos[i] = r.pos[i - 1] if i else 0.0
    rf = np.zeros(n) if rf_cal is None else np.asarray(rf_cal, float)
    return Result(r.trades, gross, r.cost, r.pos, free, free * rf)


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
        s.cost[s._row(i, j, i0 >= len(s.days[i].c))] += s.p.comm
        s.ref, s.i0, s.sg, s.ej = px, i0, sg, j

    def _row(s, i, j, at_close):
        """Строка стоимости: по close — день исполнения, внутри дня — начало его интервала."""
        return i if at_close or j == 0 else s.ix[j - 1]

    def _exit(s, i, j, b, px, reason):
        tr = s.tr
        tr.exit_day, tr.exit_bar, tr.exit_px, tr.reason, tr.fee_out = i, b, px, reason, s.p.comm
        s.gross[i] += math.log(px / s.ref)
        s.cost[s._row(i, j, reason in ("vremya", "signal"))] += s.p.comm
        s.used[i] += b + 1 - s.i0
        s.tr = s.ref = s.i0 = None

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
                return s._exit(i, j, b, d.o[b] if d.o[b] < S else S, "stop")
            if T is not None and d.h[b] > T:
                return s._exit(i, j, b, d.o[b] if d.o[b] > T else T, "take")

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
        if s.tr is not None:                        # сигнал дня входа тоже: sig[d+1] == 0 → выход по close d+1
            if s.p.tau is not None and j - s.ej >= s.p.tau and j > s.ej:
                s._exit(i, j, nb - 1, close, "vremya")
            elif s.p.exit_signal and not s.sig[i]:
                s._exit(i, j, nb - 1, close, "signal")
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
                s._exit(i, j, b, S, "stop")
            else:
                s._scan(i, j, d, b + 1)
            return
        close, nb = d.c[-1], len(d.c)
        live = bool(s.sig[i])                       # без сигнала на d+1 вход по close / повтор не ставятся
        if o["retry"] or s.p.f == "propusk":
            s.blocked = True
        elif s.p.f == "zakrytie_d1":
            if live:
                s._enter(i, j, nb - 1, close, nb, s.sigma[i], o["d_sig"])
        elif s.p.f == "povtor":
            v = s.sigma[i]
            if live and np.isfinite(v) and j + 1 < len(s.ix):
                s.order = dict(L=close * (1 - s.p.k * v), due=j + 1, retry=True, d_sig=i, sg=v)
            else:
                s.blocked = True
