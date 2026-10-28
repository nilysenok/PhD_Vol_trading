"""Ж-069 п. 5: PnL неисполненных лимиток при входе по close d+1 — синтетика с известным ответом + мутации
(testy.md п. 1, 5; mutations.md № 2). Мутации: вход по close d вместо d+1; неисполненная заявка попала в «исполненные»."""
from __future__ import annotations

import math

import numpy as np
import pytest

from fastbt import diag_zh069 as dz9
from fastbt import exec_e, exec_e_run
from fastbt.exec_e_types import Day, Params


def day(o, h, l, c, n=3):
    return Day(*[np.full(n, float(x)) for x in (o, h, l, c)])


def _run(sig, d1, d2c=101.0, comm=0.0):
    days = [day(100, 100.5, 99.5, 100), d1, day(100, 102, 96, d2c), day(100, 100.5, 99.5, 100)]
    sink: list = []
    with dz9._LOG(sink), dz9.log_counterfactual(sink):
        res = exec_e.run("AAA", days, np.array(sig), np.full(4, 0.02), Params(k=1.0, tick=0.1, comm=comm, exit_signal=True))
    return sink, res


UNF = day(100, 100.5, 98.5, 99)        # лонг: L = 98, low 98.5 — не исполнено; close d+1 = 99


def test_unfilled_long_enters_at_close_d1_known_pnl():
    (o,), _ = _run([1, 1, 0, 0], UNF)
    assert not o["isp"] and o["cf"] == dict(live=True, closed=True, pnl=pytest.approx(101 / 99 - 1))


def test_commission_both_sides_taken():
    (o,), _ = _run([1, 1, 0, 0], UNF, comm=0.0004)
    assert o["cf"]["pnl"] == pytest.approx(101 / 99 - 1 - 0.0008)


def test_unfilled_short_sign():
    (o,), _ = _run([-1, -1, 0, 0], day(100, 101.5, 99, 99), d2c=97.0)      # шорт: L = 102, high 101.5; вход 99, выход 97
    assert not o["isp"] and o["cf"]["pnl"] == pytest.approx(-(97 / 99 - 1))


def test_no_signal_on_d1_no_entry():
    (o,), _ = _run([1, 0, 0, 0], UNF)
    assert o["cf"] == dict(live=False)


def test_filled_order_has_no_counterfactual():
    (o,), _ = _run([1, 1, 0, 0], day(100, 100.5, 97.9, 99))
    assert o["isp"] and "cf" not in o


def test_open_at_end_not_counted_as_closed():
    sink: list = []
    days = [day(100, 100.5, 99.5, 100), UNF, day(100, 100.5, 99.5, 100)]
    with dz9._LOG(sink), dz9.log_counterfactual(sink):
        exec_e.run("AAA", days, np.array([1, 1, 1]), np.full(3, 0.02), Params(k=1.0, tick=0.1, comm=0.0, exit_signal=True))
    assert sink[0]["cf"]["live"] and not sink[0]["cf"]["closed"] and math.isnan(sink[0]["cf"]["pnl"])


def test_counterfactual_does_not_change_engine_result():
    sig = [1, 1, 0, 0]
    _, with_cf = _run(sig, UNF)
    days = [day(100, 100.5, 99.5, 100), UNF, day(100, 102, 96, 101.0), day(100, 100.5, 99.5, 100)]
    plain = exec_e.run("AAA", days, np.array(sig), np.full(4, 0.02), Params(k=1.0, tick=0.1, comm=0.0, exit_signal=True))
    assert np.array_equal(with_cf.gross, plain.gross) and np.array_equal(with_cf.cost, plain.cost) and len(with_cf.trades) == len(plain.trades) == 0


def test_restores_engine():
    before = exec_e_run.Run._try_order
    _run([1, 1, 0, 0], UNF)
    assert exec_e_run.Run._try_order is before


# ---------- таблица ----------
def _orders():
    return [dict(isp=True), dict(isp=False, cf=dict(live=True, closed=True, pnl=0.03)),
            dict(isp=False, cf=dict(live=True, closed=True, pnl=-0.01)), dict(isp=False, cf=dict(live=False)),
            dict(isp=False, cf=dict(live=True, closed=False, pnl=float("nan")))]


def test_row_known_answer():
    r = dz9.row("S1", [dict(pnl=0.02)], _orders())
    assert r[:3] == ("S1", 5, 1) and r[3] == pytest.approx(2.0) and r[5] == 0      # исполнено 1, закрыто 1
    assert r[6:9] == (4, 1, 2) and r[9] == pytest.approx(2.0) and r[10] == pytest.approx(1.0) and r[11] == 1
    assert r[2] + r[5] + r[6] == r[1]


def test_mutation_close_d_instead_of_d1_caught(monkeypatch):
    monkeypatch.setattr(dz9, "entry_px", lambda run, i: run.close_of(i - 1))             # вход по close d
    (o,), _ = _run([1, 1, 0, 0], UNF)
    with pytest.raises(AssertionError):
        assert o["cf"]["pnl"] == pytest.approx(101 / 99 - 1)


def test_mutation_unfilled_counted_as_filled_caught(monkeypatch):
    monkeypatch.setattr(dz9, "unfilled", lambda orders: [])                              # неисполненные пропали из своей колонки
    with pytest.raises(AssertionError):
        dz9.row("S1", [dict(pnl=0.02)], _orders())


def test_empty_tape_no_crash():
    r = dz9.row("S1", [], [])
    assert r[1] == 0 and r[6] == 0 and math.isnan(r[4]) and math.isnan(r[10])
