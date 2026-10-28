"""Ж-047, К2: правила цен и исполнения exec_e (тейк, стоп, порядок, лимит, шорт). Синтетика с известным ответом.
Каждое правило — check_*; test_* — на правильном коде, *_mut — на подложенной ошибке (check обязан упасть).
Номера мутаций — .claude/rules/mutations.md (№ 2 касание уровня, № 3 порядок в баре, № 4 гэп)."""
import math

import numpy as np
import pytest

from fastbt import exec_e_run
from fastbt.exec_e import Params, run
from test_exec_e import D, F

TK = 0.01


def rn(days, sig, sigma=0.02, close=None, **kw):
    kw.setdefault("comm", 0.0)
    return run("T", days, sig, np.full(len(days), sigma), Params(**kw), close=close)


def one(bar, sig=(1, 1, 1), **kw):
    return rn([D(F(100.0)), D([bar]), D(F(100.0))], list(sig), **kw)


# --- тейк: проход на 1 тик ---
TAKE = dict(b=1.0, R=2.0, h=1.0, tick=TK)                  # T = 100·(1 + 2·1·1·0.02) = 104


def check_take():
    assert one((100, 104.0, 99, 100), **TAKE).trades[0].reason is None          # касание — нет выхода
    t = one((100, 104.01, 99, 100), **TAKE).trades[0]                           # +1 тик — выход по уровню
    assert t.reason == "take" and t.exit_px == pytest.approx(104.0)
    t = one((105, 106, 104.5, 105), **TAKE).trades[0]                           # гэп — по open
    assert t.reason == "take" and t.exit_px == 105.0


def test_take_one_tick():
    check_take()


def test_take_touch_mutation_caught(monkeypatch):                               # № 2: «тейк по касанию»
    orig = exec_e_run.take_hit
    monkeypatch.setattr(exec_e_run, "take_hit", lambda sd, o, h, l, T, tick: orig(sd, o, h, l, T, 0.0))
    with pytest.raises(AssertionError):
        check_take()


# --- стоп: касание и гэп ---
STOP = dict(b=1.0, stop_touch=True, tick=TK)               # S = 98


def check_stop():
    t = one((100, 100, 98.0, 100), **STOP).trades[0]                            # касание — выход по S
    assert (t.reason, t.exit_px) == ("stop", pytest.approx(98.0))
    assert one((100, 100, 98.01, 100), **STOP).trades[0].reason is None
    t = one((95, 96, 94, 95), **STOP).trades[0]                                 # гэп — по open
    assert (t.reason, t.exit_px) == ("stop", 95.0)


def test_stop_touch_and_gap():
    check_stop()
    assert one((100, 100, 98.0, 100), b=1.0).trades[0].reason is None           # по умолчанию выключено: l < S


def test_stop_gap_at_level_mutation_caught(monkeypatch):                        # № 4
    orig = exec_e_run.stop_hit
    monkeypatch.setattr(exec_e_run, "stop_hit", lambda sd, o, h, l, S, t: (orig(sd, o, h, l, S, t)[0], S))
    with pytest.raises(AssertionError):
        check_stop()


def test_stop_strict_instead_of_touch_mutation_caught(monkeypatch):
    orig = exec_e_run.stop_hit
    monkeypatch.setattr(exec_e_run, "stop_hit", lambda sd, o, h, l, S, t: orig(sd, o, h, l, S, False))
    with pytest.raises(AssertionError):
        check_stop()


# --- стоп и тейк в одном баре ---
def check_order():
    r = one((100, 105, 97, 100), b=1.0, R=2.0, h=1.0, tick=TK, stop_touch=True)
    t = r.trades[0]
    assert (t.reason, t.exit_px) == ("stop", pytest.approx(98.0)) and r.gross[0] == pytest.approx(math.log(0.98))


def test_stop_first_in_same_bar():
    check_order()


def test_take_first_mutation_caught(monkeypatch):                               # № 3
    def scan(s, i, j, d, b0):
        S, T = s._levels(j)
        for b in range(b0, len(d.c)):
            o, h, l = d.o[b], d.h[b], d.l[b]
            if T is not None and exec_e_run.take_hit(1, o, h, l, T, s.p.tick)[0]:
                return s._exit(i, j, b, T, "take")
            if S is not None and exec_e_run.stop_hit(1, o, h, l, S, s.p.stop_touch)[0]:
                return s._exit(i, j, b, S, "stop")
    monkeypatch.setattr(exec_e_run.Run, "_scan", scan)
    with pytest.raises(AssertionError):
        check_order()


# --- лимит входа ---
LIM = dict(k=0.5, tick=TK)                                  # L = 99


def lim(bar, sig=(1, 1, 1), **kw):
    return rn([D(F(100.0)), D([bar]), D(F(100.0))], list(sig), **{**LIM, **kw})


def check_limit():
    r = lim((100, 100, 99.0, 100))                                              # касание — нет сделки
    assert r.trades == [] and r.n_orders == 1 and r.n_fills == 0 and r.fill_rate == 0.0
    r = lim((100, 100, 98.99, 100))                                             # на 1 тик — вход по L
    assert r.trades[0].entry_px == pytest.approx(99.0) and r.fill_rate == 1.0
    assert lim((98, 99, 97.5, 98.5)).trades[0].entry_px == 98.0                 # гэп — по open
    sig = [1, 0, 1, 1, 0]                                                       # одна из двух заявок — доля 0,5
    days = [D(F(100.0)), D(F(100.0)), D(F(100.0)), D([(100, 100, 98.9, 100)]), D(F(100.0))]
    r = rn(days, sig, **LIM)
    assert (r.n_orders, r.n_fills, r.fill_rate, len(r.trades)) == (2, 1, 0.5, 1)


def test_limit_pass_one_tick_no_fill_no_trade():
    check_limit()


def test_limit_touch_mutation_caught(monkeypatch):                              # № 2
    monkeypatch.setattr(exec_e_run, "limit_mask", lambda sd, h, l, L, tick: l <= L)
    with pytest.raises(AssertionError):
        check_limit()


# --- шорт ---
SH = dict(b=1.0, R=2.0, h=1.0, tick=TK, stop_touch=True)   # S = 102, T = 96


def check_short():
    r = one((100, 102.0, 99, 100), sig=(-1, -1, -1), **SH)                      # касание стопа
    t = r.trades[0]
    assert (t.side, t.reason, r.pos[0]) == (-1, "stop", -1.0) and t.exit_px == pytest.approx(102.0)
    assert r.gross[0] == pytest.approx(-math.log(1.02))
    assert one((100, 101, 96.0, 100), sig=(-1, -1, -1), **SH).trades[0].reason is None
    t = one((100, 101, 95.99, 100), sig=(-1, -1, -1), **SH).trades[0]           # тейк шорта: на тик ниже
    assert (t.reason, t.exit_px) == ("take", pytest.approx(96.0))
    assert lim((100, 101.0, 99, 100), sig=(-1, -1, -1)).trades == []            # лимит шорта L = 101
    assert lim((100, 101.01, 99, 100), sig=(-1, -1, -1)).trades[0].entry_px == pytest.approx(101.0)


def test_short_mirror():
    check_short()


def test_short_with_long_signs_mutation_caught(monkeypatch):
    orig = exec_e_run.stop_level
    monkeypatch.setattr(exec_e_run, "stop_level", lambda sd, *a: orig(1, *a))
    with pytest.raises(AssertionError):
        check_short()
