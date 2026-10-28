"""Ж-092 (S3/S4 п. 4), Ж-095 п. 2.3: трейлинг по m·ATR и частичная фиксация ½ на +1R (fastbt/exec_e_run.py, exec_e_rules.py).
Синтетика с известным ответом, без данных. check_* — на правильном коде; test_*_mut — подложенная ошибка (check обязан упасть):
ATR с днём d; +1R по касанию; частичный выход в баре входа; стоп ниже прежнего; комиссия частичного выхода не взята.

Лента: 14 дней разгона (одна свеча 100/101/99/100, истинный диапазон 2, ATR(14) = 2), sig = 1 с дня 13 (вход по close 100).
σ = 0,01, b = 2: начальный стоп 98, R = 2, +1R = 102."""
import numpy as np
import pytest

from fastbt import exec_e_rules, exec_e_run
from fastbt.exec_e import Params, run
from fastbt.exec_e_types import Day

COMM = 0.001


def day(rows, t0=None):
    a = np.array(rows, float).reshape(-1, 4)
    t = None if t0 is None else t0 + 5 * np.arange(len(a))
    return Day(a[:, 0], a[:, 1], a[:, 2], a[:, 3], False, t)


def flat(x, n):
    return [(x, x, x, x)] * n


def lenta(d14, d15, t0=None):
    days = [day([(100, 101, 99, 100)]) for _ in range(14)] + [day(d14, t0), day(d15)]
    sig = [0] * 13 + [1] * 3
    return days, sig


def go(d14, d15, t0=None, **kw):
    days, sig = lenta(d14, d15, t0)
    p = Params(b=2.0, comm=COMM, reentry=False, **kw)
    return run("T", days, sig, np.full(len(days), 0.01), p)


def check_trail_atr():
    """Экстремум 110, ATR(до d−1) = 2, m = 2: стоп 106; бар с low 105 выходит по 106 причиной trail."""
    d14 = [(100, 100, 100, 100), (100, 110, 100, 110), (110, 110, 105, 105)] + flat(105, 2)
    r = go(d14, flat(100, 5), atr_m=2.0)
    assert len(r.trades) == 1
    t = r.trades[0]
    assert (t.exit_day, t.exit_bar, t.reason) == (14, 2, "trail")
    assert t.exit_px == pytest.approx(106.0)
    assert r.gross.sum() == pytest.approx(np.log(1.06))
    assert r.cost.sum() == pytest.approx(2 * COMM)


def check_ratchet():
    """День 14 широкий (TR 10): ATR для дня 15 вырос до 2,571, сырой стоп 104,86 ниже вчерашнего 106 — остаётся 106."""
    d14 = [(100, 100, 100, 100), (100, 110, 100, 110)] + flat(110, 3)
    d15 = [(110, 110, 105.5, 106)] + flat(106, 4)
    r = go(d14, d15, atr_m=2.0)
    assert len(r.trades) == 1
    t = r.trades[0]
    assert (t.exit_day, t.exit_bar, t.reason) == (15, 0, "trail")
    assert t.exit_px == pytest.approx(106.0)


def check_partial():
    """+1R = 102: бар 1 (h 103) — ½ лимитом по 102; остаток — стоп 98 на следующий день. Две сделки, комиссия 0,5+0,5+1."""
    d14 = [(100, 100, 100, 100), (100, 103, 100, 103)] + flat(103, 3)
    d15 = [(103, 103, 97, 98)] + flat(98, 4)
    r = go(d14, d15, chast_1r=True)
    assert len(r.trades) == 2
    par, ch = r.trades
    assert (ch.reason, ch.parent, ch.frac, ch.exit_day, ch.exit_bar) == ("part", 0, 0.5, 14, 1)
    assert ch.exit_px == pytest.approx(102.0)
    assert ch.fee_in == 0.0 and ch.fee_out == pytest.approx(0.5 * COMM)
    assert (par.reason, par.frac, par.exit_day) == ("stop", 0.5, 15)
    assert par.exit_px == pytest.approx(98.0) and par.fee_out == pytest.approx(0.5 * COMM)
    assert r.pos[14] == pytest.approx(0.5)
    assert r.gross.sum() == pytest.approx(0.5 * np.log(1.02) + 0.5 * np.log(0.98))
    assert r.cost.sum() == pytest.approx(2 * COMM)


def check_touch():
    """high ровно на +1R (102) — не проход: ½ не закрывается (как тейк, E §2)."""
    d14 = [(100, 100, 100, 100), (100, 102, 100, 101)] + flat(101, 3)
    r = go(d14, flat(101, 5), chast_1r=True)
    assert [t.reason for t in r.trades] == [None]
    assert r.pos[14] == 1.0


def check_not_entry_bar():
    """Вход рынком по open 10:00 d+1 (bar 0, 100), в баре входа high 104 > +1R: ½ — только со следующего бара, по 103."""
    d14 = [(100, 104, 100, 103), (103, 103, 103, 103)] + flat(103, 3)
    r = go(d14, flat(103, 5), t0=600, chast_1r=True, vhod_open=True)
    ch = [t for t in r.trades if t.reason == "part"]
    assert len(ch) == 1
    assert (ch[0].exit_day, ch[0].exit_bar) == (14, 1)
    assert ch[0].exit_px == pytest.approx(103.0)


def test_trail_atr():
    check_trail_atr()


def test_ratchet():
    check_ratchet()


def test_partial():
    check_partial()


def test_touch():
    check_touch()


def test_not_entry_bar():
    check_not_entry_bar()


def test_atr_seriya():
    days = [day([(100, 101, 99, 100)]) for _ in range(15)]
    a = exec_e_rules.atr_series(days)
    assert np.isnan(a[12]) and a[13] == pytest.approx(2.0) and a[14] == pytest.approx(2.0)


def test_vyklyucheno_po_umolchaniyu():
    """Без atr_m и chast_1r результат тот же, что у прежнего пути: одна сделка, frac 1, parent None."""
    d14 = [(100, 100, 100, 100), (100, 110, 100, 110), (110, 110, 90, 95)] + flat(95, 2)
    r = go(d14, flat(95, 5))
    assert len(r.trades) == 1 and r.trades[0].frac == 1.0 and r.trades[0].parent is None
    assert r.trades[0].reason == "stop" and r.trades[0].exit_px == pytest.approx(98.0)


def test_chast_trebuet_b():
    with pytest.raises(ValueError):
        run("T", [day([(100, 101, 99, 100)])], [1], np.array([0.01]), Params(chast_1r=True))


def test_mut_atr_s_dnem_d(monkeypatch):
    monkeypatch.setattr(exec_e_run, "atr_for_day", lambda atr, ix, j: atr[ix[j]] if atr is not None else float("nan"))
    with pytest.raises(AssertionError):
        check_trail_atr()


def test_mut_1r_po_kasaniyu(monkeypatch):
    monkeypatch.setattr(exec_e_run, "part_hit", lambda sd, o, h, l, P, tick: (h >= P, max(o, P)))
    with pytest.raises(AssertionError):
        check_touch()


def test_mut_chastichnyj_v_bare_vkhoda(monkeypatch):
    monkeypatch.setattr(exec_e_run, "part_allowed", lambda day_, bar, ent: True)
    with pytest.raises(AssertionError):
        check_not_entry_bar()


def test_mut_stop_nizhe_prezhnego(monkeypatch):
    monkeypatch.setattr(exec_e_run, "ratchet", lambda sd, S, prev: S)
    with pytest.raises(AssertionError):
        check_ratchet()


def test_mut_komissiya_chastichnogo_ne_vzyata(monkeypatch):
    monkeypatch.setattr(exec_e_run, "part_fee", lambda comm, frac: 0.0)
    with pytest.raises(AssertionError):
        check_partial()
