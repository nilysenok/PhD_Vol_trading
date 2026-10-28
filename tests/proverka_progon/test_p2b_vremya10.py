"""П-2б (тестировщик): правило Ж-086 п. 4 — исполнение по времени = open 5-мин бара, не close предыдущего.
Правило: open2 = open бара, начинающегося в 10:00; 15:00 и 17:00 = open бара, начинающегося в эту минуту (сигнал на x+1 и τ);
нет бара ровно в hhmm — open первого бара позже в тот же день, нет и его — open первого бара следующего дня;
лимитный вход k_e не меняется. Тесты идут через движок (`run`), а не через вспомогательные функции: их форма — дело кодера Л-17.

Мутации (подкладывать в отдельной копии), по одной на тест test_mut_*:
1 open2 по первому бару дня (o[0]) вместо бара 10:00; 2 15:00 по close 14:55; 3 17:00 по close 16:55;
4 τ при 1500 не по бару 15:00 (close 14:55 / другой день); 5 нет бара 15:00 — берётся бар раньше вместо позже
(и то же для 10:00); 6 нет бара ≥ hhmm в дне — выход по первому бару следующего дня (09:55) вместо бара 10:00 (уточнение дирижёра: запасной
выход — open бара 10:00 следующего дня, нет его — первого бара позже 10:00); 7 вход k_e сдвинут/изменён."""
import numpy as np
import pytest

from fastbt.exec_e import Day, Params, run
from fastbt.exec_e_rules import limit_level

C = 0.0004
SIG_X = [1, 1, 0, 0, 0]                       # решение о выходе по сигналу — на close дня x = 2, исполнение в день 3


def tday(p, n=108, start=590, skip=()):
    """Бары от start (мин. от полуночи; 590 = 09:50); close = p + 0.01·i, open = close − 0.005 (open ≠ close соседнего бара)."""
    idx = [i for i in range(n) if i not in skip]
    t = np.array([start + 5 * i for i in idx])
    c = np.array([p + 0.01 * i for i in idx])
    return Day(c - 0.005, c, c - 0.005, c, False, t=t)


def go(days, sig, **kw):
    kw = dict(dict(comm=C, tau=None), **kw)
    return run("T", days, np.asarray(sig), np.full(len(days), 0.02), Params(**kw))


def lenta(day3=None, day2=None):
    days = [tday(100.0 + d) for d in range(5)]
    days[3], days[2] = day3 or days[3], day2 or days[2]
    return days


def at(day, hhmm):
    """Индекс бара, начинающегося ровно в hhmm (минуты)."""
    return int(np.flatnonzero(day.t == hhmm)[0])


def test_mut_open2_pervyj_bar_dnya():
    days = lenta()
    t = go(days, SIG_X, t_vyhod="open2").trades[0]
    assert days[3].t[0] == 590 and at(days[3], 600) == 2
    assert t.exit_px == days[3].o[2] and t.exit_px != days[3].o[0]
    days10 = [tday(100.0 + d, n=100, start=600) for d in range(5)]          # день с 10:00: бар 10:00 — первый
    assert go(days10, SIG_X, t_vyhod="open2").trades[0].exit_px == days10[3].o[0]


@pytest.mark.parametrize("tv,hhmm", [("1500", 900), ("1700", 1020)])
def test_mut_1500_1700_po_close_predydushchego_bara(tv, hhmm):
    days = lenta()
    b = at(days[3], hhmm)
    t = go(days, SIG_X, t_vyhod=tv).trades[0]
    assert days[3].t[b - 1] == hhmm - 5                                      # предыдущий бар — 14:55 / 16:55
    assert (t.exit_day, t.exit_bar, t.exit_px) == (3, b, days[3].o[b])
    assert t.exit_px != days[3].c[b - 1]


@pytest.mark.parametrize("tv,hhmm", [("1500", 900), ("1700", 1020)])
def test_mut_tau_po_baru_hhmm_dnya_x(tv, hhmm):
    days = lenta()
    b = at(days[2], hhmm)
    t = go(days, [1] * 5, tau=2, reentry=False, t_vyhod=tv).trades[0]
    assert (t.exit_day, t.exit_bar, t.reason) == (2, b, "vremya")           # день x, не x+1
    assert t.exit_px == days[2].o[b] and t.exit_px != days[2].c[b - 1]


@pytest.mark.parametrize("tv,hhmm,hole,sig", [("1500", 900, (62,), True), ("1500", 900, (62, 63), True),
                                              ("1700", 1020, (86,), True), ("open2", 600, (2,), True),
                                              ("1500", 900, (62,), False)])
def test_mut_net_bara_hhmm_beryot_bar_pozzhe(tv, hhmm, hole, sig):
    d = tday(103.0 if sig else 102.0, skip=hole)
    days = lenta(day3=d) if sig else lenta(day2=d)
    r = go(days, SIG_X if sig else [1] * 5, **({} if sig else dict(tau=2, reentry=False)), t_vyhod=tv)
    t, x = r.trades[0], (d if sig else d)
    assert x.t[t.exit_bar] == hhmm + 5 * len(hole)                           # первый бар позже hhmm, не бар раньше
    assert t.exit_px == x.o[t.exit_bar] and t.exit_day == (3 if sig else 2)


@pytest.mark.parametrize("n", [62, 40])
def test_mut_net_bara_v_dne_open_10_00_sleduyushchego(n):
    d = tday(103.0, n=n)                                                    # n=62: последний бар 14:55; n=40: 13:05
    nxt = tday(104.0, start=595)                                            # день x+2 открывается баром 09:55
    days = lenta(day3=d)
    days[4] = nxt
    t = go(days, SIG_X, t_vyhod="1500").trades[0]
    assert d.t[-1] < 900 and nxt.t[0] == 595 and nxt.t[1] == 600
    assert (t.exit_day, t.exit_bar, t.exit_px) == (4, 1, nxt.o[1])          # бар 10:00
    assert t.exit_px != nxt.o[0]                                            # мутация: первый бар следующего дня (09:55)
    assert t.exit_px != d.c[-1] and t.exit_px != d.o[-1]


def test_mut_net_bara_v_dne_i_10_00_sleduyushchego_pervyj_pozzhe_10_00():
    d = tday(103.0, n=62)
    nxt = tday(104.0, start=595, skip=(1,))                                 # нет 10:00: бары 09:55, 10:05, ...
    days = lenta(day3=d)
    days[4] = nxt
    t = go(days, SIG_X, t_vyhod="1500").trades[0]
    assert nxt.t[1] == 605 and (t.exit_day, t.exit_bar, t.exit_px) == (4, 1, nxt.o[1])
    assert t.exit_px != nxt.o[0]


def test_ke_vhod_ne_izmenilsya():
    c0, sg, tick = tday(100.0).c[-1], 0.02, 0.01
    d1 = tday(101.0)
    lo = np.array(d1.l)
    lo[5] = 99.0                                                            # лимит достижим на баре 5 дня 1
    d1 = Day(d1.o, d1.h, lo, d1.c, False, t=d1.t)
    res = []
    for tv in ("close", "1500", "1700", "open2"):
        days = [tday(100.0), d1] + [tday(102.0 + d) for d in range(3)]
        t = go(days, [1] * 5, k=0.5, zhizn=12, tick=tick, t_vyhod=tv).trades[0]
        res.append((t.entry_day, t.entry_bar, t.entry_px))
    assert res[0] == (1, 5, pytest.approx(limit_level(1, c0, 0.5, sg, tick)))
    assert all(r == res[0] for r in res)                                    # ось выхода вход не трогает
