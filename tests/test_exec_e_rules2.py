"""Ж-047, К2 (ч. 2): трейлинг, безубыток, цель R·k_s·√h, режим zakrytie и close Н6, блок повторного входа,
стоп по времени, k_s = нет, якорь. Каждое правило — check_*, правильный код проходит, мутация обязана ронять check."""
import dataclasses
import math

import numpy as np
import pytest

from fastbt import exec_e, exec_e_rules, exec_e_run
from fastbt.exec_e import Params, run
from test_exec_e import D, F
from test_exec_e_rules import one, rn


def seq(bars, n_tail=1, sig=None, **kw):
    """день 0 — F(100), далее по одному бару на день, хвост F(100)."""
    days = [D(F(100.0))] + [D([b]) for b in bars] + [D(F(100.0)) for _ in range(n_tail)]
    return rn(days, sig or [1] * len(days), **kw)


# --- трейлинг: максимум до бара − k_tr·σ ---
TR = dict(b=2.0, k_tr=1.0, stop_touch=True)                # S = 96; трейл = max high с входа · 0.98


def check_trail():
    t = seq([(100, 110, 100, 105), (108, 108, 107, 108)], **TR).trades[0]       # трейл 107.8, low 107 → выход
    assert (t.reason, t.exit_day, t.exit_px) == ("trail", 2, pytest.approx(107.8))
    assert seq([(100, 120, 101, 110)], **TR).trades[0].exit_day != 1            # внутри бара уровень не растёт (117.6 — со след. дня)
    t = seq([(100, 100, 90, 95), (91, 92, 90.5, 91)], sig=[-1] * 4, **TR).trades[0]   # шорт: min low · 1.02
    assert (t.reason, t.exit_px) == ("trail", pytest.approx(91.8))


def test_trailing_from_max_high():
    check_trail()


def test_trailing_from_close_mutation_caught(monkeypatch):
    cl = {110: 105, 120: 110, 108: 108}                    # high бара → его close
    monkeypatch.setattr(exec_e_run, "ext_next", lambda sd, e, hi, lo: max(e, cl.get(hi, hi)) if sd > 0 else min(e, lo))
    with pytest.raises(AssertionError):
        check_trail()


# --- безубыток после +1σ, со следующего бара ---
BE = dict(b=2.0, be=True, stop_touch=True)


def check_be():
    t = seq([(100, 102.5, 99.9, 101), (101, 101, 99.9, 100)], **BE).trades[0]   # +1σ в баре 1 → стоп 100 с бара 2
    assert (t.reason, t.exit_day, t.exit_px) == ("bu", 2, pytest.approx(100.0))
    assert seq([(100, 101.2, 99.9, 100.5), (101, 101, 99.9, 100)], **BE).trades[0].reason is None   # до +1σ — нет


def test_breakeven_after_one_sigma():
    check_be()


def test_breakeven_before_one_sigma_mutation_caught(monkeypatch):
    monkeypatch.setattr(exec_e_run, "be_reached", lambda sd, px, sg, hi, lo: hi >= px * (1 + 0.5 * sg))
    with pytest.raises(AssertionError):
        check_be()


# --- цель R·k_s·√h ---
def check_target():
    kw = dict(b=1.5, R=2.0, tick=0.01)                      # T = 100·(1 + 2·1.5·0.02·√4) = 112
    for extra in (dict(tau=4), dict(h=4.0)):                # h = τ (❓1) или задан
        assert seq([(100, 112.0, 99, 100)], n_tail=4, **kw, **extra).trades[0].reason != "take"
        t = seq([(100, 112.01, 99, 100)], n_tail=4, **kw, **extra).trades[0]
        assert (t.reason, t.exit_px) == ("take", pytest.approx(112.0))


def test_target_uses_sqrt_h():
    check_target()


def test_target_without_sqrt_h_mutation_caught(monkeypatch):
    monkeypatch.setattr(exec_e_rules, "take_mult", lambda p, sg: p.R * p.b * sg)
    with pytest.raises(AssertionError):
        check_target()


# --- zakrytie и close Н6 ---
def check_zakrytie():
    bar, kw = (100, 100, 97, 99), dict(b=1.0, stop_touch=True, check="zakrytie", comm=0.001, reentry=False)
    days = [D(F(100.0)), D([bar]), D(F(100.0))]
    assert rn(days, [1, 1, 1], close=[100, 99.5, 100], **kw).trades[0].reason is None   # low 97, но C_d > S
    r = rn(days, [1, 1, 1], close=[100, 97.5, 100], **kw)
    t = r.trades[0]
    assert (t.reason, t.exit_px) == ("stop", 97.5)                                  # по C Н6, не по 5-мин close 99
    assert r.gross[0] == pytest.approx(math.log(0.975)) and r.cost[1] == 0.001      # выход на close — строка дня 1
    tk = dict(b=1.0, R=2.0, h=1.0, tick=0.01, check="zakrytie")
    assert rn(days, [1, 1, 1], close=[100, 104.0, 100], **tk).trades[0].reason is None
    assert rn(days, [1, 1, 1], close=[100, 104.01, 100], **tk).trades[0].reason == "take"


def check_n6_close():
    days = [D(F(100.0)), D(F(101.0)), D(F(100.0))]
    r = rn(days, [1, 1, 1], close=[100, 101.3, 100], tau=1)
    t = r.trades[0]
    assert (t.reason, t.exit_px) == ("vremya", 101.3) and r.gross[0] == pytest.approx(math.log(1.013))


def test_zakrytie_and_n6_close():
    check_zakrytie()
    check_n6_close()


@pytest.mark.parametrize("chk", [check_zakrytie, check_n6_close])
def test_five_minute_close_mutation_caught(monkeypatch, chk):
    monkeypatch.setattr(exec_e_run.Run, "close_of", lambda s, i: s.days[i].c[-1])
    with pytest.raises(AssertionError):
        chk()


# --- блок повторного входа и стоп по времени ---
def check_block():
    days = [D(F(100.0)), D([(100, 100, 97, 100)])] + [D(F(100.0)) for _ in range(3)]
    kw = dict(b=1.0, stop_touch=True)
    r = rn(days, [1, 1, 1, 1, 1], reentry=False, **kw)
    assert len(r.trades) == 1 and list(r.pos) == [1, 0, 0, 0, 0]              # блок до новой сделки Sk
    r = rn(days, [1, 1, 0, 1, 1], reentry=False, **kw)
    assert [t.entry_day for t in r.trades] == [0, 3]
    days = [D(F(100.0 + k)) for k in range(5)]
    r = rn(days, [1] * 5, tau=2, reentry=False)                                # vremya на close d0+2, дальше блок
    assert [(t.exit_day, t.reason, t.exit_px) for t in r.trades] == [(2, "vremya", 102.0)]


def test_reentry_block_and_time_stop():
    check_block()


def test_reentry_after_stop_mutation_caught(monkeypatch):
    orig = exec_e_run.Run._exit

    def exit_(s, *a):
        orig(s, *a)
        s.blocked = 0
    monkeypatch.setattr(exec_e_run.Run, "_exit", exit_)
    with pytest.raises(AssertionError):
        check_block()


def test_time_stop_off_by_one_mutation_caught(monkeypatch):
    orig = exec_e.run
    monkeypatch.setitem(globals(), "rn", lambda d, s, **kw: orig("T", d, s, np.full(len(d), 0.02),
                        dataclasses.replace(Params(**{"comm": 0.0, **kw}), tau=kw.get("tau", 0) + 1)))
    with pytest.raises(AssertionError):
        check_block()


# --- k_s = нет: стопа нет вообще ---
def check_no_stop():
    kw = dict(k_tr=1.0, be=True, R=2.0, tau=9, tick=0.01)
    r = seq([(100, 100, 50, 50), (50, 300, 50, 60)], n_tail=1, **kw)
    assert r.trades[0].reason is None                                           # нет ни стопа, ни трейла, ни тейка


def test_no_stop_when_k_s_none():
    check_no_stop()


def test_no_stop_mutation_caught(monkeypatch):
    orig = exec_e_run.stop_level
    monkeypatch.setattr(exec_e_run, "stop_level", lambda sd, px, sg, p, e, be: orig(sd, px, sg, dataclasses.replace(p, b=1.0), e, be))
    with pytest.raises(AssertionError):
        check_no_stop()

