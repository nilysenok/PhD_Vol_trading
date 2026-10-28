"""Состояние одного прогона exec_e: вход, выходы, лимит (правила цен — exec_e_rules; учёт — docstring exec_e)."""
from __future__ import annotations

import math

import numpy as np

from fastbt.exec_e_rules import (be_reached, ext_next, limit_level, limit_mask, limit_px, stop_hit, stop_level,
                                 take_hit, take_level, tejk_level)
from fastbt.exec_e_types import Trade


class Run:
    def __init__(s, ticker, days, sig, sigma, p, close=None, div=None, nalog=1.0, tejk_M=None):
        n = len(days)
        s.tejk_M, s.tT, s.n_bez = tejk_M, None, 0     # Ж-084: M по дням сигнала; тейк сделки фиксируется на входе; входов без тейка
        s.t, s.days, s.sig, s.sigma, s.p, s.cn6 = ticker, days, sig, sigma, p, close
        s.ix = [i for i, d in enumerate(days) if not d.pusto]
        # div — полный D на шаг (день гэпа без баров — на следующий торговый); dv — D·nalog переносимой позиции (PnL);
        # sh — сдвиг стопа вниз на полный D (Ж-060: цена падает на D, стоп открытой позиции — вместе с ней)
        s.div, s.dv, s.sh, s.nal = np.zeros(n), 0.0, 0.0, float(nalog)
        if div is not None:
            for i in np.flatnonzero(np.asarray(div, float)):
                k = next((x for x in s.ix if x >= i), None)
                if k is not None:
                    s.div[k] += float(div[i])
        s.gross, s.cost, s.pos = np.zeros(n), np.zeros(n), np.zeros(n)
        s.used, s.free = np.zeros(n), np.ones(n)
        s.trades, s.tr, s.order = [], None, None
        s.ref = s.i0 = s.sg = s.ej = None
        s.blocked = 0                               # сторона, пока не сменится сигнал (0 — нет блока)
        s.ext, s.be_on, s.n_orders, s.n_fills = None, False, 0, 0
        s.xo, s.x_orders, s.x_fills = None, 0, 0    # заявка выхода {X, due, reason} (vyhod_lim, Ж-065 п. 3) и счётчики
        s.zak = p.check == "zakrytie"
        s.sr = "stop"

    def dir(s, i):
        """Сторона по сигналу дня i: Params.side · знак sig (side = −1 разворачивает сигнал 0/1 в шорт)."""
        return s.p.side * int(np.sign(s.sig[i]))

    def close_of(s, i):
        return float(s.cn6[i]) if s.cn6 is not None else s.days[i].c[-1]

    # --- вход / выход ---
    def _enter(s, i, j, b, px, i0, sg, d_sig, sd=1):
        s.tr = Trade(s.t, d_sig, i, b, px, side=sd, fee_in=s.p.comm)
        s.trades.append(s.tr)
        s.tr.row0 = s._row(i, j, i0 >= len(s.days[i].c))
        s.cost[s.tr.row0] += s.p.comm
        s.ref, s.i0, s.sg, s.ej = px, i0, sg, j
        s.ext, s.be_on, s.sh = px, False, 0.0
        if s.p.tejk != "net":                       # M зафиксирован на d_sig (≤ close d_sig), от P_вх — один раз
            s.tT = tejk_level(sd, px, float(s.tejk_M[d_sig]), s.p.tejk, s.p.tick)
            s.n_bez += s.tT is None

    def _row(s, i, j, at_close):
        """Строка стоимости: по close — день исполнения, внутри дня — начало его интервала."""
        return i if at_close or j == 0 else s.ix[j - 1]

    def _exit(s, i, j, b, px, reason):
        tr = s.tr
        tr.exit_day, tr.exit_bar, tr.exit_px, tr.reason, tr.fee_out = i, b, px, reason, s.p.comm
        s.gross[i] += tr.side * math.log((px + s.dv) / s.ref)       # dv: дивиденд позиции, пришедшей через отсечку (раз)
        s.dv = 0.0
        s.cost[s._row(i, j, s.zak or reason in ("vremya", "signal"))] += s.p.comm
        s.used[i] += b + 1 - s.i0
        if not s.p.reentry:
            s.blocked = tr.side
        s.tr = s.ref = s.i0 = None

    # --- уровни ---
    def _sg(s, j):
        sg = s.sg
        if s.p.s == "ezhedn" and j > s.ej:
            v = s.sigma[s.ix[j - 1]]
            sg = v if np.isfinite(v) else sg
        return sg if (np.isfinite(sg) and sg > 0) else None

    def _stop_shift(s, S):
        """Стоп позиции, пережившей отсечку, — на D ниже (и лонг, и шорт); без отсечек S не меняется."""
        return S if S is None or not s.sh else S - s.sh

    def _levels(s, j):
        sg = s._sg(j)
        if sg is None:
            return None, None
        tr = s.tr
        S, s.sr = stop_level(tr.side, tr.entry_px, sg, s.p, s.ext, s.be_on)    # sr: stop | trail | bu
        S = s._stop_shift(S)
        T = s.tT if s.p.tejk != "net" else take_level(tr.side, tr.entry_px, sg, s.p)
        return S, None if s.xo is not None else T    # висит заявка выхода: тейк снят (ОСО)

    def _after_bar(s, j, hi, lo):
        """Экстремум и безубыток обновляются ПОСЛЕ бара: действуют со следующего (внутри бара уровни не меняются)."""
        sd, px = s.tr.side, s.tr.entry_px
        s.ext = ext_next(sd, s.ext, hi, lo)
        sg = s._sg(j)
        if s.p.be and not s.be_on and sg is not None:
            s.be_on = be_reached(sd, px, sg, hi, lo)

    def _scan(s, i, j, d, b0):
        """Бары b0..: стоп раньше тейка; гэп — по open. Режим zakrytie — одна проверка по close Н6."""
        if s.zak:
            return s._scan_close(i, j, d)
        sd, dyn = s.tr.side, s.p.k_tr is not None or s.p.be
        S, T = s._levels(j)
        for b in range(b0, len(d.c)):
            o, h, l = d.o[b], d.h[b], d.l[b]
            if S is not None:
                hit, px = stop_hit(sd, o, h, l, S, s.p.stop_touch)
                if hit:
                    return s._exit(i, j, b, px, s.sr)
            if T is not None:
                hit, px = take_hit(sd, o, h, l, T, s.p.tick)
                if hit:
                    return s._exit(i, j, b, px, "take")
            if s.xo is not None and limit_mask(-sd, h, l, s.xo["X"], s.p.tick):     # после стопа: только проход на тик
                s.x_fills += 1
                return s._exit(i, j, b, limit_px(-sd, o, s.xo["X"]), "lim")
            if dyn:
                s._after_bar(j, h, l)
                S, T = s._levels(j)

    def _scan_close(s, i, j, d):
        C, nb, sd = s.close_of(i), len(d.c), s.tr.side
        S, T = s._levels(j)
        if S is not None and stop_hit(sd, C, C, C, S, s.p.stop_touch)[0]:
            return s._exit(i, j, nb - 1, C, s.sr)
        if T is not None and take_hit(sd, C, C, C, T, s.p.tick)[0]:
            return s._exit(i, j, nb - 1, C, "take")
        s._after_bar(j, C, C)

    # --- день ---
    def step(s, j, i):
        d, nb = s.days[i], len(s.days[i].c)
        s.dv = s.div[i] * s.nal if s.tr is not None else 0.0   # позиция с прошлого close пересекает отсечку: лонг получит D, шорт заплатит
        if s.tr is not None:
            s.i0 = 0
            s.sh += s.div[i]
        if s.order is not None and s.order["due"] == j:
            s._try_order(i, j, d)
        elif s.tr is not None:
            s._scan(i, j, d, 0)
        close = s.close_of(i)
        if s.xo is not None:                        # заявка выхода не исполнилась за d+1: выход по close с прежней причиной
            if s.tr is not None:
                s._exit(i, j, nb - 1, close, s.xo["reason"])
            s.xo = None
        if s.tr is not None:                        # сигнал дня входа тоже: sig[d+1] != стороне → выход по close d+1
            if s.p.tau is not None and j - s.ej >= s.p.tau and j > s.ej:
                s._leave(i, j, nb, close, "vremya")
            elif s.p.exit_signal and s.dir(i) != s.tr.side:
                s._leave(i, j, nb, close, "signal")
        if s.tr is not None:
            s.gross[i] += s.tr.side * math.log((close + s.dv) / s.ref)
            s.dv = 0.0
            s.ref = close
            s.used[i] += nb - s.i0
            s.i0 = nb
        if s.blocked and s.dir(i) != s.blocked:
            s.blocked = 0
        s._decide(i, j, close, nb)
        s.pos[i] = float(s.tr.side) if s.tr is not None else 0.0
        s.free[i] = 1.0 - s.used[i] / nb

    def _leave(s, i, j, nb, close, reason):
        """Выход по signal/vremya: рынком по close дня решения; при vyhod_lim — лимитная заявка на d+1 (due = j+1, не раньше, Д2)
        на X = limit_level(−сторона, close, k, σ̂_d, тик). Нет d+1 в окне или σ̂ нечисловая — как раньше, по close."""
        sg = s.sigma[i]
        if s.p.vyhod_lim and j + 1 < len(s.ix) and np.isfinite(sg):
            s.x_orders += 1
            s.xo = dict(X=limit_level(-s.tr.side, close, s.p.k, sg, s.p.tick), due=j + 1, reason=reason)
        else:
            s._exit(i, j, nb - 1, close, reason)

    def _place(s, close, sd, sg, j, i, **kw):
        s.n_orders += 1
        s.order = dict(L=limit_level(sd, close, s.p.k, sg, s.p.tick), due=j + 1, sd=sd, d_sig=i, **kw)

    def _decide(s, i, j, close, nb):
        sg, sd = s.sigma[i], s.dir(i)
        if s.tr is not None or s.order is not None or s.blocked or not sd or not np.isfinite(sg):
            return
        if s.p.k is None:
            s._enter(i, j, nb - 1, close, nb, sg, i, sd)
        elif j + 1 < len(s.ix):
            s._place(close, sd, sg, j, i, retry=False)

    def _try_order(s, i, j, d):
        o, s.order = s.order, None
        sd = o["sd"]
        hit = np.flatnonzero(limit_mask(sd, d.h, d.l, o["L"], s.p.tick))
        if hit.size:
            b = int(hit[0])
            px = limit_px(sd, d.o[0] if s.zak else d.o[b], o["L"])
            s.n_fills += 1
            s._enter(i, j, b, px, b, s.sigma[o["d_sig"]] if not o["retry"] else o["sgo"], o["d_sig"], sd)
            S, _ = s._levels(j)
            if not s.zak and S is not None and stop_hit(sd, d.o[b], d.h[b], d.l[b], S, s.p.stop_touch)[0]:
                s._exit(i, j, b, S, s.sr)
            else:
                s._scan(i, j, d, b + 1)
            return
        close, nb = s.close_of(i), len(d.c)
        live = bool(s.dir(i) == sd)        # без сигнала на d+1 вход по close / повтор не ставятся
        if o["retry"] or s.p.f == "propusk":
            s.blocked = sd
        elif s.p.f == "zakrytie_d1":
            if live:
                s._enter(i, j, nb - 1, close, nb, s.sigma[i], o["d_sig"], sd)
        elif s.p.f == "povtor":
            v = s.sigma[i]
            if live and np.isfinite(v) and j + 1 < len(s.ix):
                s._place(close, sd, v, j, i, retry=True, sgo=v)
            else:
                s.blocked = sd
