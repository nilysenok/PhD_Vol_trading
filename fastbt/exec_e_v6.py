"""V6-b (Ж-111 пп. 7, 8б, 10; Ж-114 п. 5): безубыток b0, «стоп в плюс», ступенчатый трейлинг, пирамидинг.

Чистые функции + миксин для Run (exec_e_run). Всё выключено по умолчанию (Params), тогда путь прежний побайтно.
Все уровни действуют со следующего бара (как be, k_tr); добавка пирамидинга — как лимит: только проход сквозь уровень,
не касание, и строго после бара входа. Добавки — отдельные ноги Trade(reason "pir", parent, frac = размер, своя комиссия).
"""
from __future__ import annotations

import dataclasses
import math

from fastbt.exec_e_rules import EPS, be_reached, part_allowed, ratchet, take_hit
from fastbt.exec_e_types import Trade

PIR_MAX = 2          # Ж-111 п. 10: не больше двух добавок
RISK_MAX = 2.0       # общий риск ставки ≤ 2ρ (ρ = риск базовой ноги = расстояние до начального стопа)


def v6_on(p) -> bool:
    return p.b0 is not None or p.b0_c is not None or p.sp is not None or p.step_s is not None or p.pir > 0


def v6_check(p, rang, n_days: int) -> None:
    if not v6_on(p):
        return
    if p.b is None or p.check != "bary" or p.vhod_tip != "podtverzhdenie":
        raise ValueError("b0 / sp / step_s / pir требуют b, check='bary' и vhod_tip='podtverzhdenie'")
    if sum(x is not None for x in (p.b0, p.b0_c)) + bool(p.be) > 1:
        raise ValueError("be, b0 и b0_c взаимоисключающие")
    if (p.b0 is not None and not p.b0 > 0) or (p.b0_c is not None and not p.b0_c > 0):
        raise ValueError("b0 и b0_c должны быть > 0")
    if p.b0_c is not None and (rang is None or len(rang) != n_days):
        raise ValueError("b0_c требует rang длины days (ранг силы на день сигнала, [0; 1])")
    if p.sp is not None and not (len(p.sp) == 2 and p.sp[0] > 0 and 0 <= p.sp[1] < p.sp[0]):
        raise ValueError("sp = (p, q): p > 0, 0 ≤ q < p")
    if p.step_s is not None and not p.step_s > 0:
        raise ValueError("step_s должен быть > 0")
    if not 0 <= p.pir <= 1:
        raise ValueError("pir: доля размера добавки в [0; 1]")


def v5_new(p) -> bool:
    """Оси Л6 включены — медленный путь (exec_e._new_axes → fast_etap1._slow_only)."""
    return p.k_e != 0 or p.a_rang is not None or p.chast is not None


def v5_check(p, rang, n_days: int) -> None:
    if p.a_rang is not None:
        if p.a is not None or p.R is not None or len(p.a_rang) != 2 or not p.a_rang[0] > 0 or not p.a_rang[1] >= 0:
            raise ValueError("a_rang = (a0 > 0, a1 ≥ 0), без a и R: добавка за силу не против позиции")
        if rang is None or len(rang) != n_days:
            raise ValueError("a_rang требует rang длины days (ранг силы на день сигнала, [0; 1])")
    if p.chast is not None:
        f, r = p.chast if len(p.chast) == 2 else (0, 0)
        if p.chast_1r or not 0 < f < 1 or not r > 0 or p.b is None or p.check != "bary":
            raise ValueError("chast = (доля в (0; 1), r > 0): без chast_1r, нужен b, check='bary'")
    if p.k_e != 0 and (p.vhod_tip != "stop_zayavka" or not math.isfinite(p.k_e)):
        raise ValueError("k_e ≠ 0 — только при vhod_tip = stop_zayavka")


def take_params(p, rang_d):
    """Параметры тейка сделки: a = a0 + a1·ранг дня сигнала. Ранг нечисловой → False (тейка нет). a_rang нет — p как есть."""
    if p.a_rang is None:
        return p
    r = float(rang_d)
    return dataclasses.replace(p, a=p.a_rang[0] + p.a_rang[1] * r) if math.isfinite(r) else False


def chast_pair(p):
    """(доля, r) частичного выхода или None: chast_1r ≡ (½, 1)."""
    return (0.5, 1.0) if p.chast_1r else p.chast


def be_mult(p, rang_d) -> float | None:
    """Порог перевода в безубыток, в σ: b0, c·ранг, 1 (be) или None. Ранг нечисловой → nan (порог недостижим)."""
    if p.b0 is not None:
        return p.b0
    if p.b0_c is not None:
        r = float(rang_d)
        return p.b0_c * r if math.isfinite(r) else float("nan")
    return 1.0 if p.be else None


def step_level(sd, px, sg, st, ext, b):
    """Ступенчатый трейлинг: n = целых шагов st·σ экстремума от входа; стоп = начальный + n·st·σ·P. n = 0 — None."""
    n = math.floor(sd * (ext - px) / (px * st * sg) + EPS)
    return px * (1 - sd * b * sg) + sd * n * st * sg * px if n > 0 else None


def plus_level(sd, px, sg, q):
    return px * (1 + sd * q * sg)


def pir_level(sd, px, sg, m):
    """Уровень m-й добавки: +m·σ от цены входа базовой ноги (m = 1, 2) — только по движению в сторону позиции."""
    return px * (1 + sd * m * sg)


def pir_size(sd, p_add, S, legs, r0, a):
    """Размер добавки: a, урезанный так, чтобы сумма риска ног до текущего стопа S не превысила RISK_MAX·r0.
    legs — [(цена, размер)] действующих ног; риск ноги = размер·max(0, сторона·(цена − S))."""
    risk = sum(w * max(0.0, sd * (px - S)) for px, w in legs)
    room = RISK_MAX * r0 - risk
    if room <= 0:
        return 0.0
    dr = sd * (p_add - S)
    return a if dr <= 0 else min(a, room / dr)


class V6:
    """Состояние и действия V6-b; подмешивается в Run. Поля: be_m, pq_on, vlev, adds [(цена, размер, индекс сделки)], n_add."""

    def v6_enter(s, d_sig):
        s.be_m = be_mult(s.p, s.rang[d_sig] if s.rang is not None else float("nan"))
        s.pq_on, s.vlev, s.adds, s.n_add = False, None, [], 0
        s.p_take = take_params(s.p, s.rang[d_sig] if s.rang is not None else float("nan"))   # ранг дня сигнала d_sig, не дня входа

    def v6_extra(s, sg):
        """Кандидаты стопа: «в плюс» и ступенчатый; не ослабляется (ratchet); сдвиг на D делает _stop_shift."""
        p, tr, c = s.p, s.tr, []
        if p.sp is not None and s.pq_on:
            c.append(plus_level(tr.side, tr.entry_px, sg, p.sp[1]))
        if p.step_s is not None:
            x = step_level(tr.side, tr.entry_px, sg, p.step_s, s.ext, p.b)
            if x is not None:
                c.append(x)
        if c:
            s.vlev = ratchet(tr.side, max(c) if tr.side > 0 else min(c), s.vlev)
        return [] if s.vlev is None else [(s.vlev, "trail")]

    def v6_after(s, sg, hi, lo):
        sd, px = s.tr.side, s.tr.entry_px
        if s.p.sp is not None and not s.pq_on:
            s.pq_on = be_reached(sd, px, s.p.sp[0] * sg, hi, lo)
        if s.be_m is not None and not s.be_on:
            s.be_on = be_reached(sd, px, s.be_m * sg, hi, lo)

    def v6_pir(s, i, j, b, o, h, l, S, sg):
        """Добавка по тренду: проход сквозь +m·σ (m = n_add + 1), цена max(open, уровень) (шорт зеркально); одна на бар; нужен стоп S."""
        tr = s.tr
        if S is None or s.n_add >= PIR_MAX or not part_allowed(i, b, s.ent):
            return
        hit, px = take_hit(tr.side, o, h, l, pir_level(tr.side, tr.entry_px, sg, s.n_add + 1), s.p.tick)
        if not hit:
            return
        legs = [(tr.entry_px, s.w - sum(a for _, a, _ in s.adds))] + [(x, a) for x, a, _ in s.adds]
        a = pir_size(tr.side, px, S, legs, s.R, s.p.pir)
        if a > 1e-12:
            s._add(i, j, b, px, a)

    def _add(s, i, j, b, px, a):
        tr = s.tr
        s.gross[i] += tr.side * s.w * math.log((px + s.dv) / s.ref)      # прежняя позиция до цены добавки; D — ей одной
        s.dv = 0.0
        s.used[i] += (b + 1 - s.i0) * s.w
        row = s._row(i, j, False)
        s.cost[row] += s.p.comm * a
        s.trades.append(Trade(s.t, tr.d_sig, i, b, px, reason="pir", side=tr.side, fee_in=s.p.comm * a, frac=a,
                              parent=s.trades.index(tr), row0=row))
        s.adds.append((px, a, len(s.trades) - 1))
        s.ref, s.i0, s.w, s.n_add = px, b + 1, s.w + a, s.n_add + 1

    def v6_exit(s, i, b, px) -> float:
        """Ноги добавок закрываются вместе с родителем: бар, цена, своя комиссия выхода. Возвращает сумму их размеров."""
        for _, a, k in s.adds:
            t = s.trades[k]
            t.exit_day, t.exit_bar, t.exit_px, t.fee_out = i, b, px, s.p.comm * a
        tot = sum(a for _, a, _ in s.adds)
        s.adds = []
        return tot
