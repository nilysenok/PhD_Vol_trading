"""Л6.3 куба v7 (Ж-146 п. 6, Ж-153 п. 1): вход в T того же дня — Params.vhod_T = 'ЧЧ:ММ' (медленный путь; быстрое ядро — лист V7-6).

Сигнал дня i посчитан по состоянию на T (t_reshenie_v7.td_T); вход — в тот же день по цене `cena[i]` = t_reshenie_v7.cena_T
(open 10-мин бара, начинающегося в T; для 18:40 — close этого бара, аукцион). Позиция открывается на баре T (для 18:40 — на
последнем 5-мин баре 10-мин бара), стоп и тейк — со следующего бара; в 18:40 баров после входа нет. Нет 5-мин бара T в данных
или cena[i] нечисловая — сделки в этот день нет (без подстановки, Д4); сигнал живёт, решение — на следующий день.
Если в этот день позицию закрыл выход по сигналу/сроку на close (позже T), входа нет: вход в T раньше выхода — перекрытие во времени.
Комиссия, дивиденды, плата за шорт, конец окна — как у Run (exec_e_run): тот же _enter, _exit, step (Д7)."""
from __future__ import annotations

import math

import numpy as np

from fastbt.exec_e2 import Run2
from fastbt.t_reshenie_v7 import AUKCION, _minuty

STEP10 = 10                               # длина бара цены T, минуты


def bar_vhoda(day, t: int) -> int | None:
    """Индекс 5-мин бара входа в T (минуты от полуночи): бар, начинающийся ровно в t; для аукциона — последний бар 10-мин
    бара [t; t+10). Нет такого бара — None. Day.t нет — ValueError (время бара не восстановить)."""
    tt = getattr(day, "t", None)
    if tt is None:
        raise ValueError("vhod_T требует Day.t (начало 5-мин баров)")
    tt = np.asarray(tt)
    k = int(np.searchsorted(tt, t, side="left"))
    if k >= len(tt) or tt[k] != t:
        return None
    return int(np.searchsorted(tt, t + STEP10, side="left")) - 1 if t == AUKCION else k


def proverka(p, cena, n_days: int, urovni) -> None:
    """Параметры оси и ряд цен T: vhod_T ⇔ cena_T, длина days; совместимость — вход рынком в T, не лимит и не стоп-заявка."""
    if p.vhod_T is None:
        if cena is not None:
            raise ValueError("cena_T — только при Params.vhod_T")
        return
    _minuty(p.vhod_T)
    if cena is None or len(cena) != n_days:
        raise ValueError("vhod_T требует cena_T длины days (t_reshenie_v7.cena_T по дням)")
    if (p.k is not None or p.vhod_open or p.T != 1 or p.vhod_tip != "podtverzhdenie" or p.reentry_den or p.vyhod_lim
            or p.check != "bary" or p.bar_plus or urovni is not None):
        raise ValueError("vhod_T — вход рынком в T (k = None), check='bary', без vhod_open, T > 1, stop_zayavka, reentry_den, vyhod_lim, bar_plus")


class RunT(Run2):
    def __init__(s, *a, cena=None, **kw):
        super().__init__(*a, **kw)
        s.cena, s.tm = np.asarray(cena, float), _minuty(s.p.vhod_T)

    def _decide(s, i, j, close, nb):
        sg, sd = s.sigma[i], s.dir(i)
        if s.tr is not None or s.order is not None or s.blocked or not sd or not np.isfinite(sg):
            return
        if s.last_x is not None and s.last_x.exit_day == i:      # выход на close дня i позже T: входа в T нет
            return
        d, px = s.days[i], s.cena[i]
        b = bar_vhoda(d, s.tm)
        if b is None or not math.isfinite(px):
            return
        s._enter(i, j, b, float(px), b, sg, i, sd)
        S, _ = s._levels(j)
        if not s._stop_vhoda(i, j, d, b, float(px), S):
            s._scan(i, j, d, b + 1)
        if s.tr is not None:                                     # позиция дожила до close: оценка дня — как в step
            c = s.close_of(i)
            s.gross[i] += s.tr.side * s.w * math.log(c / s.ref)
            s.ref, s.used[i], s.i0 = c, s.used[i] + (nb - s.i0) * s.w, nb
