"""Состояние одного прогона exec_e: вход, выходы, лимит (правила цен — exec_e_rules; учёт — docstring exec_e)."""
from __future__ import annotations

import math

import numpy as np

from fastbt.exec_e_parab import pb_after, pb_level, pb_start
from fastbt.exec_e_rules import (atr_for_day, be_reached, ext_next, limit_level, limit_mask, limit_px, part_allowed,
                                 part_fee, part_hit, part_level, ratchet, rodnoy_level, stop_hit, stop_level,
                                 take_hit, take_level, tejk_level, v_bare)
from fastbt.exec_e_piv import enter_levels, piv_check, piv_on
from fastbt.exec_e_types import Trade
from fastbt import exec_e_minuderzh as minuderzh
from fastbt.exec_e_v6 import V6, chast_pair, v5_check, v6_check, v6_on
from fastbt.exec_e_zayavki import prodlit, reenter


class Run(V6):
    mu = mu_raz = None                              # V7-4 (exec_e_minuderzh): отложенный выход (сделка, причина) / отложенный разворот в T (сделка)
    stale = frozenset()                           # Д5-2: шаги j после разрыва > Params.razryv дней (заполняет exec_e.run); пусто — прежнее поведение

    def __init__(s, ticker, days, sig, sigma, p, close=None, div=None, nalog=1.0, tejk_M=None, atr=None, rang=None, us=None, ut=None):
        n = len(days)
        s.us, s.ut = us, ut                         # Л5: ряды единицы стопа/безубытка и тейка по дню сигнала; None — σ̂ (путь прежний)
        v6_check(p, rang, n)
        v5_check(p, rang, n)                        # Л6: k_e, a_rang, chast
        if p.time_decay and (p.tau is None or p.tejk != "net"):
            raise ValueError("time_decay: нужен tau и tejk = net (тейк a·σ или R — половина дистанции после tau//2 дней)")
        piv_check(p, tejk_M, n)                     # V6-10: piv_sl, piv_tp, vwap_m; уровни — PivUrovni в tejk_M
        s.lv, s.pv = (tejk_M if piv_on(p) else None), (None, None)    # (стоп, тейк) по уровням — на входе
        s.cf, s.p_take = chast_pair(p), p          # частичный выход (доля, r); параметры тейка сделки
        s.rang, s.be_m, s.pq_on, s.vlev, s.adds, s.n_add, s.v6 = rang, None, False, None, [], 0, v6_on(p)   # V6-b (exec_e_v6)
        s.atr, s.w, s.smax, s.part, s.R, s.ent = atr, 1.0, None, False, float("nan"), (-1, -1)   # Ж-092: ATR, остаток, ratchet, ½, R, бар входа
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
        s.last_x, s.done_re = None, -1              # V6-a: последняя закрытая сделка; день последнего повторного входа
        s.vne = []                                  # исполнения вне [low; high] бара (инвариант _v_bare): обязан быть пуст

    def dir(s, i):
        """Сторона по сигналу дня i: Params.side · знак sig (side = −1 разворачивает сигнал 0/1 в шорт)."""
        return s.p.side * int(np.sign(s.sig[i]))

    def close_of(s, i):
        return float(s.cn6[i]) if s.cn6 is not None else s.days[i].c[-1]

    # --- вход / выход ---
    def _v_bare(s, i, b, px):
        """Инвариант: цена исполнения — в [low; high] бара b дня i; нарушение — ValueError (тикер, день, бар, цена, [low; high]).
        Close дня (Н6) — цена рынка по определению; zakrytie и bar_plus берут цену другого бара того же дня — по диапазону дня."""
        d = s.days[i]
        den = s.zak or getattr(s.p, "bar_plus", 0)
        lo, hi = (min(d.l), max(d.h)) if den else (d.l[b], d.h[b])
        if px != s.close_of(i) and not v_bare(lo, hi, px):
            s.vne.append((i, b, px))
            raise ValueError(f"исполнение вне [low; high] бара: {s.t} день {i} бар {b} цена {px} [{lo}; {hi}]")

    def _stop_vhoda(s, i, j, d, b, px, S):
        """Стоп в баре входа активен с момента исполнения по px: «уже пробитый» (лонг S ≥ px, шорт S ≤ px) — выход по px,
        как гэп по open (БЛОКЕР #596; прежде — по S, цены могло не быть в баре). Иначе — по S. True — позиция закрыта."""
        if S is None:
            return False
        hit, xp = stop_hit(s.tr.side, px, d.h[b], d.l[b], S, s.p.stop_touch)
        if hit:
            s._exit(i, j, b, xp, s.sr)
        return hit

    def _enter(s, i, j, b, px, i0, sg, d_sig, sd=1):
        s._v_bare(i, min(b, len(s.days[i].c) - 1), px)
        s.tr = Trade(s.t, d_sig, i, b, px, side=sd, fee_in=s.p.comm)
        s.trades.append(s.tr)
        s.tr.row0 = s._row(i, j, i0 >= len(s.days[i].c))
        s.cost[s.tr.row0] += s.p.comm
        s.ref, s.i0, s.sg, s.ej = px, i0, sg if s.us is None else s.us[d_sig], j
        s.sgt = sg if s.ut is None else s.ut[d_sig]
        s.ext, s.be_on, s.sh = px, False, 0.0
        s.w, s.smax, s.part, s.ent = 1.0, None, False, (i, b)
        s.v6_enter(d_sig)
        if s.lv is not None:                        # V6-10: уровни строки d_sig (≤ close d_sig), от цены входа; фиксируются
            s.pv = enter_levels(s.p, s.lv, sd, px, d_sig, s.close_of(d_sig), s.p.tick)
        s.pb, s.pb_new = (pb_start(sd, px, s.sg, s.p, s.days[i].h[b], s.days[i].l[b]), True) if s.p.parab is not None else (None, False)   # П3b
        s.R =px * s.p.b * s.sg if s.p.b is not None else float("nan")
        if s.p.tejk in ("k_srednemu", "pol_puti"):  # M зафиксирован на d_sig (≤ close d_sig), от P_вх — один раз
            s.tT = tejk_level(sd, px, float(s.tejk_M[d_sig]), s.p.tejk, s.p.tick)
            s.n_bez += s.tT is None

    def _row(s, i, j, at_close):
        """Строка стоимости: по close — день исполнения, внутри дня — начало его интервала."""
        return i if at_close or j == 0 else s.ix[j - 1]

    def _exit(s, i, j, b, px, reason):
        s._v_bare(i, b, px)
        tr = s.last_x = s.tr
        tr.exit_day, tr.exit_bar, tr.exit_px, tr.reason, tr.fee_out = i, b, px, reason, s.p.comm * (s.w - s.v6_exit(i, b, px))
        s.gross[i] += tr.side * s.w * math.log((px + s.dv) / s.ref)       # dv: дивиденд позиции, пришедшей через отсечку (раз)
        s.dv = 0.0
        s.cost[s._row(i, j, s.zak or reason in ("vremya", "signal"))] += s.p.comm * s.w
        s.used[i] += (b + 1 - s.i0) * s.w
        if not s.p.reentry:
            s.blocked = tr.side
        s.tr = s.ref = s.i0 = None

    def _partial(s, i, j, b, px):
        """Ж-092: ½ позиции лимитом на +1R — отдельная сделка (reason "part", parent — индекс родителя). Остаток живёт дальше."""
        s._v_bare(i, b, px)
        tr, FRAC = s.tr, s.cf[0]
        fee = part_fee(s.p.comm, FRAC)
        s.trades.append(Trade(s.t, tr.d_sig, tr.entry_day, tr.entry_bar, tr.entry_px, exit_day=i, exit_bar=b, exit_px=px,
                              reason="part", side=tr.side, fee_in=0.0, fee_out=fee, frac=FRAC,
                              parent=s.trades.index(tr), row0=tr.row0))
        s.gross[i] += tr.side * FRAC * math.log((px + s.dv) / s.ref)       # dv достаётся и закрытой, и оставшейся половине
        s.cost[s._row(i, j, False)] += fee
        s.used[i] += (b + 1 - s.i0) * s.w
        s.i0, s.w, s.part, tr.frac = b + 1, s.w - FRAC, True, 1.0 - FRAC

    # --- уровни ---
    def _sg(s, j, take=False):
        sg, arr = (s.sgt, s.ut) if take else (s.sg, s.us)
        if s.p.s == "ezhedn" and j > s.ej:
            v = (s.sigma if arr is None else arr)[s.ix[j - 1]]
            sg = v if np.isfinite(v) else sg
        return sg if (np.isfinite(sg) and sg > 0) else None

    def _chand(s, j):
        """Л5 Chandelier (vyhody_v5_medl, tt<2): экстремум 11 торговых дней ДО дня j ∓ k·ATR(14) на конец дня j−1; нет окна или ATR — кандидата нет."""
        a, w, sd = atr_for_day(s.atr, s.ix, j), s.ix[max(0, j - 11):j], s.tr.side
        if not w or not math.isfinite(a):
            return []
        ex = max(max(s.days[i].h) for i in w) if sd > 0 else min(min(s.days[i].l) for i in w)
        return [(ex - sd * s.p.chand * a, "trail")]

    def _stop_shift(s, S):
        """Стоп позиции, пережившей отсечку, — на D ниже (и лонг, и шорт); без отсечек S не меняется."""
        return S if S is None or not s.sh else S - s.sh

    def _levels(s, j):
        sg = s._sg(j)
        if sg is None:
            return None, None
        tr = s.tr
        ex = s.v6_extra(sg) if s.v6 else []         # V6-b: «стоп в плюс», ступенчатый; выкл — вызов прежний
        ex = ex + s._chand(j) if s.p.chand is not None else ex
        kx = dict(extra=ex) if ex else {}
        if s.p.atr_m is None and s.p.chand is None:
            S, s.sr = stop_level(tr.side, tr.entry_px, sg, s.p, s.ext, s.be_on, **kx)    # sr: stop | trail | bu
        else:                                       # Ж-092: ATR по дням до текущего; стоп трейла не ослабляется (ATR может падать)
            S, s.sr = stop_level(tr.side, tr.entry_px, sg, s.p, s.ext, s.be_on, atr_for_day(s.atr, s.ix, j), **kx)
            S = s.smax = ratchet(tr.side, S, s.smax)
        if s.pb is not None:                        # П3b: в баре входа — начальный стоп; далее по состоянию на конец прошлого бара
            S, s.sr = (s.pb[0] if s.pb_new else pb_level(tr.side, s.pb)), "trail"
        S, s.sr = (s.pv[0], "stop") if s.pv[0] is not None else (S, s.sr)    # V6-10: стоп пивота / VWAP (единственный)
        S = s._stop_shift(S)
        if s.p.tejk == "rodnoy_uroven":
            T = s._rodnoy(j)
        else:
            st = s._sg(j, True)
            st = st * s._td(j) if st is not None else st
            T = s.tT if s.p.tejk != "net" else (take_level(tr.side, tr.entry_px, st, s.p_take) if st is not None and s.p_take is not False else None)
        T = s.pv[1] if s.pv[1] is not None else T    # V6-10: тейк пивота
        return S, None if s.xo is not None else T    # висит заявка выхода: тейк снят (ОСО)

    def _td(s, j):
        """V6-9: множитель дистанции тейка — ½ при time_decay, когда в позиции больше tau//2 торговых дней (j − ej > tau//2), иначе 1."""
        return 0.5 if s.p.time_decay and j - s.ej > s.p.tau // 2 else 1.0

    def _rodnoy(s, j):
        """Ж-091: заявка дня ix[j] стоит на L[ix[j−1]] (данные ≤ 18:50 прошлого торгового дня), обновляется каждый вечер;
        активна с первого бара дня; tejk_M — [2, n]: строка 0 лонг, строка 1 шорт."""
        if j == 0:
            return None
        sd = s.tr.side
        return rodnoy_level(sd, float(s.tejk_M[0 if sd > 0 else 1][s.ix[j - 1]]), s.p.tick)

    def _after_bar(s, j, hi, lo):
        """Экстремум и безубыток обновляются ПОСЛЕ бара: действуют со следующего (внутри бара уровни не меняются)."""
        sd, px = s.tr.side, s.tr.entry_px
        if s.pb is not None:
            pb_after(sd, s.pb, s.p, hi, lo)
        s.ext = ext_next(sd, s.ext, hi, lo)
        sg = s._sg(j)
        if s.v6 and sg is not None:
            s.v6_after(sg, hi, lo)
        elif s.p.be and not s.be_on and sg is not None:
            s.be_on = be_reached(sd, px, sg, hi, lo)

    def _scan(s, i, j, d, b0):
        """Бары b0..: стоп раньше тейка; гэп — по open. Режим zakrytie — одна проверка по close Н6."""
        if s.zak:
            return s._scan_close(i, j, d)
        sd, dyn = s.tr.side, s.p.k_tr is not None or s.p.be or s.p.atr_m is not None or s.v6 or s.pb is not None
        s.pb_new = False                            # П3b: бар входа пройден
        S, T = s._levels(j)
        for b in range(b0, len(d.c)):
            o, h, l = d.o[b], d.h[b], d.l[b]
            if S is not None:
                hit, px = stop_hit(sd, o, h, l, S, s.p.stop_touch)
                if hit:
                    return s._exit(i, j, b, px, s.sr)
            if s.cf is not None and not s.part and part_allowed(i, b, s.ent):     # после стопа, до тейка; не в баре входа
                P = part_level(sd, s.tr.entry_px, s.R * s.cf[1], s.p.tick)
                hit, px = part_hit(sd, o, h, l, P, s.p.tick)
                if hit:
                    s._partial(i, j, b, px)
            if T is not None:
                hit, px = take_hit(sd, o, h, l, T, s.p.tick)
                if hit:
                    return s._exit(i, j, b, px, "take")
            if s.xo is not None and limit_mask(-sd, h, l, s.xo["X"], s.p.tick):     # после стопа: только проход на тик
                s.x_fills += 1
                return s._exit(i, j, b, limit_px(-sd, o, s.xo["X"]), "lim")
            if s.p.pir and s._sg(j) is not None:    # V6-b: добавка после стопа, тейка и выхода-заявки (худший порядок)
                s.v6_pir(i, j, b, o, h, l, S, s._sg(j))
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
        if j in s.stale:                            # Д5-2: заявка (вход) через остановку торгов снята; позиция держится, повторный вход — вечером j−1 устарел
            s.order = None
        if s.order is not None and s.order["due"] == j:
            s._try_order(i, j, d)
        elif s.tr is not None:
            s._scan(i, j, d, 0)
        if s.p.reentry_den and j not in s.stale:    # V6-a: повторный вход в тот же день после внутридневного выхода
            reenter(s, i, j, d)
        close = s.close_of(i)
        if s.xo is not None:                        # заявка выхода не исполнилась за d+1: выход по close с прежней причиной
            if s.tr is not None:
                s._exit(i, j, nb - 1, close, s.xo["reason"])
            s.xo = None
        if s.tr is not None:                        # V7-4: min_uderzhanie — до N-го дня выход откладывается (reshenie помнит), с N-го обязателен
            why = minuderzh.reshenie(s, i, j, s.p.tau is not None and j - s.ej >= s.p.tau and j > s.ej)
            if why:
                s._leave(i, j, nb, close, why)
        if s.tr is not None:
            s.gross[i] += s.tr.side * s.w * math.log((close + s.dv) / s.ref)
            s.dv = 0.0
            s.ref = close
            s.used[i] += (nb - s.i0) * s.w
            s.i0 = nb
        if s.blocked and s.dir(i) != s.blocked:
            s.blocked = 0
        s._decide(i, j, close, nb)
        s.pos[i] = float(s.tr.side) * s.w if s.tr is not None else 0.0
        s.free[i] = 1.0 - s.used[i] / nb

    def _sig_exit(s, i):
        """Сигнал ушёл от стороны сделки. rodnoy_uroven — сам родной сигнальный выход, выставленный заявкой: рыночного нет (Ж-091)."""
        return s.p.exit_signal and s.p.tejk != "rodnoy_uroven" and s.dir(i) != s.tr.side

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
        if s.p.k is None and s.p.vhod_open:         # Ж-086 п. 1: рынком по open 10:00 дня d+1 (заявка без уровня)
            if j + 1 < len(s.ix):
                s.order = dict(L=None, due=j + 1, sd=sd, d_sig=i, rynok=True)
        elif s.p.k is None:
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
            if s.zak or not s._stop_vhoda(i, j, d, b, px, S):
                s._scan(i, j, d, b + 1)
            return
        if prodlit(s, o, i, j):                     # V6-a: T > 1 — заявка живёт, пока сигнал активен
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
