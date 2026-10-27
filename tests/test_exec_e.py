"""Семья E, движок исполнения (fastbt/exec_e.py): синтетика с известным ответом. Номера — мутации §8."""
import math

import numpy as np
import pytest

from fastbt.exec_e import Day, Params, run
from fastbt.stats import pos_from_trades

C = 0.0004
LN = math.log


def D(rows, pusto=False):
    a = np.array(rows, float).reshape(-1, 4) if rows else np.zeros((0, 4))
    return Day(a[:, 0], a[:, 1], a[:, 2], a[:, 3], pusto)


def F(x, n=1):
    return [(x, x, x, x)] * n


def go(days, sig, sig_hat, **kw):
    sg = np.full(len(days), sig_hat) if np.isscalar(sig_hat) else np.asarray(sig_hat, float)
    return run("T", days, sig, sg, Params(**kw))


# --- якорь: вход по close d без уровней = дневной движок A ---
def test_anchor_equals_daily_A():
    rng = np.random.default_rng(7)
    n = 60
    cl = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    days = [D([(c, c * 1.01, c * 0.99, c)] + F(c)) for c in cl]
    sig = (rng.random(n) < 0.5).astype(int)
    r = go(days, sig, 0.02, k=None, comm=C)
    te, tx, e = [], [], 0
    while e < n:                                    # сигнальные сделки A: [e, x), x — первый sig == 0
        if not sig[e]:
            e += 1
            continue
        x = e + 1
        while x < n and sig[x]:
            x += 1
        te.append(e)
        tx.append(x)
        e = x
    pos = pos_from_trades(np.array(te), np.array(tx), np.ones(len(te), int), len(te), n)
    lr = np.r_[np.diff(np.log(cl)), 0.0]
    cost_a = np.abs(np.diff(pos, prepend=0.0)) * C
    assert np.array_equal(r.pos, pos)
    assert np.allclose(r.gross[1:], (pos * lr)[:-1], atol=1e-12, rtol=0)
    assert np.allclose(r.cost, cost_a, atol=1e-12, rtol=0)
    assert [t.entry_day for t in r.trades] == te
    assert [t.exit_day for t in r.trades if t.exit_day is not None] == [x for x in tx if x < n]
    assert all(t.reason == "signal" for t in r.trades if t.exit_day is not None)
    assert all(t.entry_px == cl[t.entry_day] for t in r.trades)


# --- лимит ---
def L0(k=0.5, s=0.02, p=100.0):
    return p * (1 - k * s)


def test_p2_fill_only_on_bars_after_d():
    L = L0()
    days = [D(F(100.0) + [(100, 100, 90, 100)]), D(F(100.0, 3)), D(F(100.0, 3))]   # d: low ниже L, d+1: нет
    r = go(days, [1, 0, 0], 0.02, k=0.5, comm=C)
    assert r.trades == []
    days[1] = D(F(100.0) + [(100, 100, 98.5, 100)] + F(100.0))
    r = go(days, [1, 1, 0], 0.02, k=0.5, comm=C)
    t = r.trades[0]
    assert (t.entry_day, t.entry_bar, t.entry_px, t.d_sig) == (1, 1, L, 0)


def test_p3_touch_is_not_fill():
    L = L0()
    days = [D(F(100.0)), D([(100, 100, L, 100)]), D(F(100.0))]
    assert go(days, [1, 0, 0], 0.02, k=0.5).trades == []
    days[1] = D([(100, 100, L - 1e-9, 100)])
    assert go(days, [1, 0, 0], 0.02, k=0.5).trades[0].entry_px == L


def test_gap_below_limit_fills_at_open():
    days = [D(F(100.0)), D([(97.0, 98.0, 96.0, 97.5)]), D(F(100.0))]
    t = go(days, [1, 0, 0], 0.02, k=0.5).trades[0]
    assert t.entry_px == 97.0


def test_entry_and_stop_in_same_bar():
    L = L0()
    S = L * (1 - 0.02)
    days = [D(F(100.0)), D([(100, 100, 95.0, 100)]), D(F(100.0))]
    r = go(days, [1, 1, 1], 0.02, k=0.5, b=1.0, comm=C)
    t = r.trades[0]
    assert (t.reason, t.entry_bar, t.exit_bar, t.exit_day) == ("stop", 0, 0, 1)
    assert t.exit_px == S
    assert r.gross[1] == pytest.approx(LN(S / L), abs=1e-12)


# --- f ---
def f_days():
    return [D(F(100.0)), D(F(100.0, 2)), D(F(100.0) + [(100, 100, 97.9, 99.2)]), D(F(100.0, 2))]


def test_f_propusk():
    assert go(f_days(), [1, 0, 0, 0], 0.02, k=0.5, f="propusk").trades == []


def test_f_zakrytie_d1():
    t = go(f_days(), [1, 1, 0, 0], 0.02, k=0.5, f="zakrytie_d1").trades[0]
    assert (t.entry_day, t.entry_px, t.entry_bar) == (1, 100.0, 1)


def test_f_povtor_uses_new_sigma_and_only_once():
    sg = [0.02, 0.04, 0.02, 0.02]                      # L2 = 100·(1 − 0.5·0.04) = 98
    r = go(f_days(), [1, 1, 0, 0], sg, k=0.5, f="povtor")
    t = r.trades[0]
    assert (t.entry_day, t.entry_px, t.d_sig) == (2, 98.0, 1)
    days = [D(F(100.0)), D(F(100.0)), D(F(100.0)), D(F(100.0))]
    assert go(days, [1, 1, 1, 1], 0.02, k=0.5, f="povtor").trades == []   # повтор один


# --- тейк / стоп ---
def ts_days(bar):
    return [D(F(100.0)), D([bar]), D(F(100.0))]


def test_p4_stop_before_take_in_same_bar():
    r = go(ts_days((100, 103, 97, 100)), [1, 1, 1], 0.02, a=1.0, b=1.0, comm=C)
    t = r.trades[0]
    assert (t.reason, t.exit_px) == ("stop", 98.0)
    assert r.pnl.sum() == pytest.approx(LN(0.98) - 2 * C, abs=1e-12) and r.pnl.sum() < 0


def test_take_and_gap_up_take():
    t = go(ts_days((100, 103, 99, 100)), [1, 1, 1], 0.02, a=1.0, b=1.0).trades[0]
    assert (t.reason, t.exit_px) == ("take", 102.0)
    t = go(ts_days((105, 106, 104, 105)), [1, 1, 1], 0.02, a=1.0, b=1.0).trades[0]
    assert (t.reason, t.exit_px) == ("take", 105.0)


def test_p5_gap_through_stop_at_open():
    r = go(ts_days((95.0, 96.0, 94.0, 95.0)), [1, 1, 1], 0.02, a=1.0, b=1.0)
    assert r.trades[0].exit_px == 95.0 and r.trades[0].reason == "stop"
    assert r.gross[1] == pytest.approx(LN(0.95), abs=1e-12)


def test_p9_exit_fee_on_intraday_stop():
    r = go(ts_days((100, 100, 97, 100)), [1, 1, 1], 0.02, b=1.0, comm=C)
    assert r.cost[0] == C and r.cost[1] == C            # вход на close 0, выход по стопу в день 1
    assert r.trades[0].fee_in == C and r.trades[0].fee_out == C


# --- s, τ ---
def test_s_fiks_vs_ezhedn():
    days = [D(F(100.0)), D(F(100.0)), D([(100, 100, 98.5, 100)]), D(F(100.0))]
    sg = [0.02, 0.01, 0.02, 0.02]                       # день 2 ezhedn: σ̂ закрытия дня 1 → S = 99
    fx = go(days, [1, 1, 1, 1], sg, b=1.0, s="fiks")
    ez = go(days, [1, 1, 1, 1], sg, b=1.0, s="ezhedn")
    assert fx.trades[0].reason is None or fx.trades[0].reason != "stop"
    assert (ez.trades[0].reason, ez.trades[0].exit_px, ez.trades[0].exit_day) == ("stop", 99.0, 2)


def test_tau_exit_at_close_of_entry_plus_tau():
    days = [D(F(100.0)), D(F(101.0)), D(F(102.0)), D(F(103.0))]
    r = go(days, [1, 1, 1, 1], 0.02, tau=2)
    t = r.trades[0]
    assert (t.reason, t.exit_day, t.exit_px) == ("vremya", 2, 102.0)


# --- пустые бары ---
def test_pusto_day_carries_position_and_return_lands_after_pause():
    days = [D(F(100.0)), D([], pusto=True), D(F(110.0)), D(F(110.0))]
    r = go(days, [1, 1, 1, 0], 0.02)
    assert list(r.pos) == [1, 1, 1, 0]
    assert r.gross[1] == 0 and r.gross[2] == pytest.approx(LN(1.1), abs=1e-12)
    assert r.free[1] == 0.0


def test_pusto_day_does_not_fill_order_and_not_counted_in_tau():
    L = L0()
    days = [D(F(100.0)), D([], pusto=True), D(F(100.0) + [(100, 100, L - 1, 100)]), D(F(100.0)), D(F(100.0))]
    t = go(days, [1, 1, 1, 1, 1], 0.02, k=0.5, tau=1).trades[0]
    assert (t.entry_day, t.exit_day) == (2, 3)


# --- ставка ---
def test_p10_free_capital_inside_day_gets_rate():
    L = L0()
    bars = F(100.0, 3) + [(100, 100, 98.0, 100)] + F(100.0, 2) + [(100, 100, 97.0, 100)] + F(100.0, 3)
    days = [D(F(100.0)), D(bars), D(F(100.0))]
    r = run("T", days, [1, 1, 1], np.full(3, 0.02), Params(k=0.5, b=1.0, comm=C),
            rf_cal=[0.001, 0.002, 0.001])
    t = r.trades[0]
    assert (t.entry_bar, t.exit_bar, t.exit_px) == (3, 6, L * 0.98)
    assert r.free[1] == pytest.approx(1 - 4 / 10)       # в рынке бары 3..6 из 10
    assert r.fdr[1] == pytest.approx(0.6 * 0.002)
    assert r.pos[1] == 0.0 and r.free[0] == 1.0 and r.free[2] == 1.0
