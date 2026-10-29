"""К-wf5b (Ж-137 п. 6): при t_vyhod=open2 выход по сигналу — open x+1, выход по сроку τ — close дня истечения.

Ручная лента, ответ известен. Бары 09:50…18:45 (индекс 10:00 = 2). Вход по close дня 0 (сигнал 1), τ = 2 дня.
Мутация «τ по open x+1»: vyhod_plan("open2", "vremya") → (1, 600) — тесты ниже обязаны падать.
1500/1700: по τ — open бара hhmm дня истечения (срок известен заранее; выход внутри дня x, Д2 не нарушена).
"""
import numpy as np
import pytest

from fastbt import exec_e_vremya
from fastbt.exec_e import Params, run
from fastbt.exec_e_types import Day

N = 108


def day(o=100.0, c=100.0, hi=None, lo=None, over=None):
    t = np.arange(590, 590 + 5 * N, 5)
    a = np.full((N, 4), 100.0)
    a[:, 0] = o
    a[:, 3] = c
    a[:, 1] = hi if hi is not None else max(o, c)
    a[:, 2] = lo if lo is not None else min(o, c)
    for b, v in (over or {}).items():
        a[b] = v
    return Day(a[:, 0], a[:, 1], a[:, 2], a[:, 3], False, t)


def rn(days, sig, **kw):
    kw.setdefault("comm", 0.0)
    kw.setdefault("check", "bary")
    return run("T", days, list(sig), np.full(len(days), 0.05), Params(**kw))


def tr(r):
    t = r.trades[0]
    return t.exit_day, t.exit_px, t.reason


def lenta():
    return [day(100, 100), day(101, 101.5), day(102, 103, hi=103.2, lo=101.8), day(110, 111, hi=111.5, lo=109.5),
            day(120, 121, hi=121.5, lo=119.5)]


def test_tau_open2_close_dnya_istecheniya():
    # вход close дня 0 = 100; τ=2 истекает на дне 2 → close дня 2 = 103 (не open дня 3 = 110)
    assert tr(rn(lenta(), (1, 1, 1, 1, 1), t_vyhod="open2", tau=2)) == (2, 103.0, "vremya")


def test_signal_open2_open_x_plus_1():
    # сигнал снят на дне 1 (решение по close дня 1) → выход по open дня 2 = 102
    assert tr(rn(lenta(), (1, 0, 0, 0, 0), t_vyhod="open2")) == (2, 102.0, "signal")


def test_signal_i_tau_v_odin_den_vyigryvaet_srok():
    # сигнал снят на дне 2 и τ истекает на дне 2: close дня 2 (103) раньше open дня 3 (110) → причина vremya
    assert tr(rn(lenta(), (1, 1, 0, 0, 0), t_vyhod="open2", tau=2)) == (2, 103.0, "vremya")


def test_signal_ranshe_sroka_ostayotsya_open_x_plus_1():
    # сигнал снят на дне 1, срок τ=3 истёк бы позже: выход по сигналу open дня 2
    assert tr(rn(lenta(), (1, 0, 0, 0, 0), t_vyhod="open2", tau=3)) == (2, 102.0, "signal")


def test_close_put_pobaytno_prezhniy():
    # t_vyhod=close: τ → close дня 2, сигнал → close дня 1 (поведение до К-wf5b не менялось)
    assert tr(rn(lenta(), (1, 1, 1, 1, 1), t_vyhod="close", tau=2)) == (2, 103.0, "vremya")
    assert tr(rn(lenta(), (1, 0, 0, 0, 0), t_vyhod="close")) == (1, 101.5, "signal")


def test_tau_open2_v_diapazone_low_high():
    r = rn(lenta(), (1, 1, 1, 1, 1), t_vyhod="open2", tau=2)
    d = lenta()[r.trades[0].exit_day]
    assert d.l.min() <= r.trades[0].exit_px <= d.h.max()


@pytest.mark.parametrize("tv,bar,px", [("1500", 62, 103.0), ("1700", 86, 103.0)])
def test_tau_1500_1700_open_bara_dnya_istecheniya(tv, bar, px):
    # срок известен заранее: по τ выход внутри дня истечения (день 2) по open бара hhmm, не на x+1
    days = lenta()
    days[2] = day(102, 103, hi=103.2, lo=101.8, over={bar: (103.0, 103.1, 102.9, 103.0)})
    r = rn(days, (1, 1, 1, 1, 1), t_vyhod=tv, tau=2)
    assert tr(r) == (2, px, "vremya") and r.trades[0].exit_bar == bar


def test_mutatsiya_tau_po_open_x_plus_1_lovitsya(monkeypatch):
    orig = exec_e_vremya.vyhod_plan
    monkeypatch.setattr(exec_e_vremya, "vyhod_plan", lambda tv, pr: (1, 600) if tv == "open2" else orig(tv, pr))
    # на мутации ответ другой (open дня 3 = 110): test_tau_open2_close_dnya_istecheniya упадёт
    assert tr(rn(lenta(), (1, 1, 1, 1, 1), t_vyhod="open2", tau=2)) == (3, 110.0, "vremya")


def test_bystryj_sloy_ne_znaet_open2():
    from fastbt import fast_exec as fx
    assert "t_vyhod" in fx.ne_znaet(Params(t_vyhod="open2", check="bary"))
    assert "t_vyhod" not in fx.ne_znaet(Params(t_vyhod="close", check="bary"))
