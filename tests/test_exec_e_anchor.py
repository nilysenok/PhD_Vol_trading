"""Семья E: якорь к A через msp.portfolio (r, fdr, excess), сигнал d+1, повторный вход, края."""
import numpy as np
import pandas as pd
import pytest

from fastbt import msp
from fastbt.exec_e import run, Params
from fastbt.stats import pos_from_trades
from test_exec_e import C, D, F, L0, go


def a_trades(sig):
    """Сделки сигнального A: [e, x), x — первый sig == 0 после входа на close e."""
    n, te, tx, e = len(sig), [], [], 0
    while e < n:
        if not sig[e]:
            e += 1
            continue
        x = e + 1
        while x < n and sig[x]:
            x += 1
        te.append(e)
        tx.append(x)
        e = x
    return te, tx


@pytest.mark.parametrize("seed", [7, 11])
def test_anchor_equals_A_msp_portfolio(seed):
    rng = np.random.default_rng(seed)
    n = 80
    cl = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    days = [D([(c, c * 1.01, c * 0.99, c)] + F(c)) for c in cl]
    sig = (rng.random(n) < 0.5).astype(int)
    sig[-1] = 0                                     # A платит выход на последней строке окна (Н5)
    dates = pd.bdate_range("2021-03-01", periods=n)
    rate = pd.Series([0.07, 0.085], pd.to_datetime(["2021-01-01", "2021-05-10"]))
    rf_cal = rate.reindex(dates, method="ffill").values / 365.0 * msp.cal_days(dates)
    r = run("T", days, sig, np.full(n, 0.02), Params(k=None, comm=C), rf_cal=rf_cal)
    te, tx = a_trades(sig)
    pos = pos_from_trades(np.array(te), np.array(tx), np.ones(len(te), int), len(te), n)
    lr = np.r_[np.diff(np.log(cl)), 0.0]
    port = msp.portfolio(pd.DataFrame(dict(date=dates, ticker="T", strategy="S", position=pos,
                                           daily_gross_return=pos * lr)), rate, C)
    tol = dict(atol=1e-12, rtol=0)
    assert np.array_equal(r.pos, pos) and np.array_equal(r.pos, port["exposure"].values)
    assert np.allclose(r.pnl, port["r"].values, **tol)
    assert np.allclose(r.fdr, port["fdr"].values, **tol)
    assert np.allclose(r.pnl + r.fdr - rf_cal, port["excess"].values, **tol)
    assert [t.entry_day for t in r.trades] == te and [t.exit_day for t in r.trades] == tx


def test_no_rate_on_day_of_close_entry():
    days = [D(F(100.0)), D(F(100.0)), D(F(100.0))]
    r = run("T", days, [1, 1, 0], np.full(3, 0.02), Params(), rf_cal=[1.0, 1.0, 1.0])
    assert list(r.fdr) == [0.0, 0.0, 1.0]           # вход по close 0: ставки за день 0 нет; выход close 2: с дня 2


# --- сигнал d+1 (правило: sig[d+1] == 0 → позиция не доживает до d+2) ---
def f_days(fill_d1):
    d1 = F(100.0) + [(100, 100, 98.5, 99.5)] if fill_d1 else F(100.0, 2)
    return [D(F(100.0)), D(d1), D(F(100.0) + [(100, 100, 97.0, 99.0)])]


@pytest.mark.parametrize("f", ["propusk", "zakrytie_d1", "povtor"])
def test_signal_gone_on_d1_after_limit_fill_exits_at_close_d1(f):
    r = go(f_days(True), [1, 0, 0], 0.02, k=0.5, f=f)
    t = r.trades[0]
    assert (t.entry_px, t.exit_day, t.reason) == (L0(), 1, "signal") and r.pos[1] == 0.0


def test_signal_gone_on_d1_unfilled_zakrytie_and_povtor_do_not_enter():
    for f in ("zakrytie_d1", "povtor"):
        assert go(f_days(False), [1, 0, 0], 0.02, k=0.5, f=f).trades == []
    t = go(f_days(False), [1, 1, 0], 0.02, k=0.5, f="zakrytie_d1").trades[0]
    assert (t.entry_day, t.exit_day) == (1, 2)
    t = go(f_days(False), [1, 1, 0], 0.02, k=0.5, f="povtor").trades[0]
    assert (t.entry_day, t.exit_day, t.reason) == (2, 2, "signal")


# --- повторный вход как у A: на баре выхода при живом сигнале ---
@pytest.mark.parametrize("bar,reason", [((100, 100, 97, 98), "stop"), ((100, 103, 100, 102), "take")])
def test_reentry_at_exit_close_after_stop_and_take(bar, reason):
    days = [D(F(100.0)), D([bar]), D(F(bar[3]))]
    r = go(days, [1, 1, 0], 0.02, a=1.0, b=1.0)
    assert [t.reason for t in r.trades] == [reason, "signal"]
    assert (r.trades[1].entry_day, r.trades[1].entry_px) == (1, bar[3])


def test_reentry_after_time_stop():
    days = [D(F(100.0)), D(F(101.0)), D(F(102.0)), D(F(103.0))]
    r = go(days, [1, 1, 1, 0], 0.02, tau=1)
    assert [(t.entry_day, t.exit_day, t.reason) for t in r.trades] == [
        (0, 1, "vremya"), (1, 2, "vremya"), (2, 3, "vremya")]


# --- края ---
def test_touch_stop_and_take_do_not_execute():
    S, T = 100 * (1 - 0.02), 100 * (1 + 0.02)
    days = [D(F(100.0)), D([(100, T, S, 100)]), D(F(100.0))]
    t = go(days, [1, 1, 1], 0.02, a=1.0, b=1.0).trades[0]
    assert t.reason is None and t.exit_day is None   # low == S, high == T: сделка открыта


def test_pause_inside_hold_does_not_count_for_tau():
    days = [D(F(100.0)), D([], pusto=True), D(F(101.0)), D([], pusto=True), D(F(102.0)), D(F(103.0))]
    r = go(days, [1] * 6, 0.02, tau=2)
    t = r.trades[0]
    assert (t.exit_day, t.exit_px, t.reason) == (4, 102.0, "vremya")
    assert list(r.pos) == [1] * 6                    # позиция переносится через паузы; на close 4 — повторный вход


@pytest.mark.parametrize("bad", [[(100, 100, 100, np.nan)], [(100, 100, 100, 100), (100, 100, 100, np.nan)], []])
def test_bad_bars_raise_valueerror(bad):
    with pytest.raises(ValueError, match="день 1"):
        go([D(F(100.0)), D(bad), D(F(100.0))], [1, 1, 1], 0.02)


def test_nan_sigma_means_no_order_and_length_mismatch_raises():
    assert go([D(F(100.0))] * 3, [1, 1, 1], np.nan, k=0.5).trades == []
    with pytest.raises(ValueError):
        go([D(F(100.0))] * 3, [1, 1], 0.02)
