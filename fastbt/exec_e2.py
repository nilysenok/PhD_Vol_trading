"""Run2(Run): новые оси исполнения прогона этапа 1 (контракт docs/otchety/2026-10-28_progon_etap1.md §1.1, Л-2).

ed (массив единиц лимита дня сигнала), zhizn, gap_propusk, t_vyhod, bar_plus, stop_tik. Без них exec_e.run берёт прежний Run.
Правила — чистые функции ниже (мутации в tests/test_exec_e2.py подменяют именно их).
Время баров — fastbt/exec_e_vremya (Л-3): bar_do(day, hhmm) -> int | None, vyhod_plan(t_vyhod, prichina) -> (сдвиг дня, hhmm | None).
t_vyhod ≠ close: выход по signal/vremya переносится: сдвиг 1 — pend на день x+1 (бары до bar_do — как обычно, стоп первым; на баре
bar_do — выход по c[b]; open2 — по o[0]); сдвиг 0 (τ в 15:00/17:00 дня x) — срез дня на bar_do. Нет бара — выход по close Н6, флаг прежний.
Комиссия такого выхода — в строке начала интервала. Бар bar_do сканируется включительно, стоп первым (решение дирижёра, класс B).
"""
from __future__ import annotations

import math

import numpy as np

from fastbt.exec_e_rules import limit_level, limit_mask
from fastbt.exec_e_run import Run
from fastbt.exec_e_types import Day

_LATE = ("stop", "trail", "bu", "take", "lim")      # причины внутридневного исполнения, к которым применим bar_plus


def _vr():
    from fastbt import exec_e_vremya
    return exec_e_vremya


def gap_over(sd, o0, L, tick) -> bool:
    """open бара 0 уже за лимитом: лонг o₀ ≤ L − tick (тик не задан — строго o₀ < L); шорт зеркально."""
    if tick is None:
        return o0 < L if sd > 0 else o0 > L
    return o0 <= L - tick if sd > 0 else o0 >= L + tick


def zhizn_hits(hit, n):
    """Бары исполнения лимита, не позже жизни заявки: b < n (n = None — весь день)."""
    return hit if n is None else hit[hit < n]


def later_px(sell, level, o_n, c6):
    """Цена исполнения на следующем баре: open, если хуже уровня; иначе уровень. Бара нет (конец дня) — close Н6."""
    if o_n is None:
        return c6
    return min(level, o_n) if sell else max(level, o_n)


def scan_end(cut):
    """Скан строго до бара выхода (Ж-086 п. 4): стоп и тейк — по барам до него; выход — по open бара cut."""
    return cut


def tik_stop(sd, S, tick, n):
    """Стоп хуже на n тиков: лонг ниже, шорт выше."""
    return S - sd * n * tick


class Run2(Run):
    def __init__(s, *a, ed=None, **kw):
        super().__init__(*a, **kw)
        s.ed, s.pend, s.x_t = ed, None, 0

    # --- лимит ---
    def _place(s, close, sd, sg, j, i, **kw):
        u = sg if s.ed is None else s.ed[i]             # единица смещения лимита; стоп, тейк, размер — по σ̂
        if not np.isfinite(u):
            return
        s.n_orders += 1
        s.order = dict(L=limit_level(sd, close, s.p.k, u, s.p.tick), due=j + 1, sd=sd, d_sig=i, **kw)

    def _try_rynok(s, i, j, d):
        """Ж-086 п. 1: вход рынком по open бара 10:00 (bar_at: нет бара — первый позже; после 10:00 баров нет — open 10:00
        следующего дня; времени баров нет — open первого бара). Стоп и тейк — с самого бара входа, после его open."""
        vr = _vr()
        o, s.order = s.order, None
        b = vr.bar_at(d, vr.HHMM["open2"])
        b = 0 if b is None else b
        if b >= len(d.c):
            s.order = dict(o, due=j + 1) if j + 1 < len(s.ix) else None
            return
        s._enter(i, j, b, float(d.o[b]), b, s.sigma[o["d_sig"]], o["d_sig"], o["sd"])
        s._scan(i, j, d, b)

    def _try_order(s, i, j, d):
        if s.order.get("rynok"):                        # вход рынком по open 10:00 d+1: без уровня, жизни заявки и gap
            return s._try_rynok(i, j, d)
        o = s.order
        sd, tick = o["sd"], s.p.tick
        if s.p.gap_propusk and gap_over(sd, d.o[0], o["L"], tick):
            s.order, s.blocked = None, sd
            return
        hit = zhizn_hits(np.flatnonzero(limit_mask(sd, d.h, d.l, o["L"], tick)), s.p.zhizn)
        if not hit.size:                                # за жизнь заявки не исполнилась: как «не дошла до уровня»
            s.order = dict(o, L=-math.inf if sd > 0 else math.inf)
        super()._try_order(i, j, d)

    # --- bar_plus, stop_tik ---
    def _bp(s, i, b, level, sell):
        n, d = s.p.bar_plus, s.days[i]
        return later_px(sell, level, d.o[b + n] if b + n < len(d.c) else None, s.close_of(i))

    def _enter(s, i, j, b, px, i0, sg, d_sig, sd=1):
        if s.p.bar_plus and i0 == b and b < len(s.days[i].c):          # лимитный вход внутри дня (вход по close: i0 = nb)
            px = s._bp(i, b, px, sd < 0)
        super()._enter(i, j, b, px, i0, sg, d_sig, sd)

    def _levels(s, j):
        S, T = super()._levels(j)
        if S is not None and s.p.stop_tik:
            S = tik_stop(s.tr.side, S, s.p.tick, s.p.stop_tik)
        return S, T

    def _exit(s, i, j, b, px, reason, row_start=False):
        if s.p.bar_plus and reason in _LATE:
            px = s._bp(i, b, px, s.tr.side > 0)
        if not row_start:
            return super()._exit(i, j, b, px, reason)
        tr = s.tr
        super()._exit(i, j, b, px, "stop")              # комиссия — в строку начала интервала (как внутридневной выход)
        tr.reason = reason

    def _exit_t(s, i, j, b, px, reason):
        s.x_t += 1
        s._exit(i, j, b, px, reason, row_start=True)

    # --- t_vyhod ---
    def _leave(s, i, j, nb, close, reason):
        if s.pend is not None:                          # выход уже перенесён на open следующего дня (нет бара после hhmm)
            return
        if s.p.t_vyhod != "close":
            sdv, hh = _vr().vyhod_plan(s.p.t_vyhod, reason)
            if sdv and j + 1 < len(s.ix):
                s.pend = dict(reason=reason, due=j + 1, hhmm=hh)
                return
        super()._leave(i, j, nb, close, reason)

    def _scan(s, i, j, d, b0):
        pe = s.pend if s.pend and s.pend["due"] == j else None
        cut = reason = None
        if pe:
            s.pend, reason = None, pe["reason"]
            cut = _vr().bar_at(d, pe["hhmm"])
        elif s.p.t_vyhod in ("1500", "1700") and s.p.tau is not None and j > s.ej and j - s.ej >= s.p.tau:
            sdv, hh = _vr().vyhod_plan(s.p.t_vyhod, "vremya")
            if not sdv:
                cut, reason = _vr().bar_at(d, hh), "vremya"
        if cut is None:
            super()._scan(i, j, d, b0)
            if pe and s.tr is not None:                 # времени баров нет: выход по close Н6 этого дня
                s._exit(i, j, len(d.c) - 1, s.close_of(i), reason)
            return
        e = scan_end(cut)
        if b0 < e:
            super()._scan(i, j, Day(d.o[:e], d.h[:e], d.l[:e], d.c[:e]), b0)
        if s.tr is None:
            return
        if cut < len(d.c):
            return s._exit_t(i, j, cut, d.o[cut], reason)   # Ж-086 п. 4: open бара hhmm (или первого после него)
        if j + 1 < len(s.ix):                           # после hhmm баров нет: open бара 10:00 следующего дня (как open2)
            s.pend = dict(reason=reason, due=j + 1, hhmm=_vr().HHMM["open2"])
        else:
            s._exit(i, j, len(d.c) - 1, s.close_of(i), reason)
