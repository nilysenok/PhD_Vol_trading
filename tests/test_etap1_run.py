"""К4 Ж-047: run_e1 (строки g, sides, expo с весом на сделку и границами окна), port_rows, gate_trades.
Только синтетика; мутация — подмена функции, тест обязан упасть AssertionError (mutations.md № 5, № 10)."""
import numpy as np
import pandas as pd
import pytest

from fastbt import etap1_run, exec_e
from fastbt.etap1_data import gate_trades
from fastbt.exec_e_types import Day, Params
from fastbt.s9 import Engine

C = 0.0004


def day1(c):
    a = np.array([c], float)
    return Day(a, a, a, a, False)


def anchor_ref(pos, lr, a, b):
    """Что считает Engine.port для окна [a; b) одной пары: g, sides (вход на первой, выход на последней строке), |pos|."""
    p = pos[a:b]
    dp = np.diff(p, prepend=0.0)
    last = np.r_[np.zeros(len(p) - 1, bool), True]
    return p * lr[a:b], np.abs(dp) + np.where(last, np.abs(p), 0.0), np.abs(p)


def check_anchor(runner, seed, n=40):
    rng = np.random.default_rng(seed)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    pos = rng.choice([-1.0, 0.0, 1.0], n, p=[0.25, 0.35, 0.4])
    a = int(rng.integers(0, 10))
    b = n - 1
    lr = np.log(px[1:] / px[:-1])
    sig = pos.copy()
    sig[b] = 0.0
    r = runner("T", [day1(x) for x in px], sig, np.ones(n), Params(check="bary"), w=None, win=(a, b))
    g, sides, expo = anchor_ref(pos, np.r_[lr, 0.0], a, b)
    assert np.allclose(r.g, g, rtol=0, atol=1e-15), "g ≠ pos·lr"
    assert np.allclose(r.sides, sides, rtol=0, atol=1e-15), "стороны окна ≠ правилу Engine.port"
    assert np.allclose(r.expo, expo, rtol=0, atol=1e-15), "expo ≠ |pos|"


@pytest.mark.parametrize("seed", range(60))
def test_anchor_equals_engine_port_rule_random_windows(seed):
    check_anchor(exec_e.run_e1, seed)


def test_mutation_first_row_not_clipped_caught():
    """Мутация: вход переносимой в окно сделки не платится в строке a (стороны первой строки окна → 0)."""
    def bad(*a, **kw):
        r = etap1_run.run_e1(*a, **kw)
        r.sides[0] = 0.0
        return r
    with pytest.raises(AssertionError):
        for seed in range(60):
            check_anchor(bad, seed)


def test_mutation_last_row_exit_not_paid_caught():
    def bad(*a, **kw):
        r = etap1_run.run_e1(*a, **kw)
        r.sides[-1] -= np.minimum(r.sides[-1], np.abs(r.expo[-1]))
        return r
    with pytest.raises(AssertionError):
        for seed in range(60):
            check_anchor(bad, seed)


def test_weight_fixed_per_trade_with_limit_entry_row():
    """Лимитный вход на d+1 внутри дня: строка входа — d (начало интервала); вес = вес дня сигнала, не вход-дня."""
    o = [100, 100, 101, 102, 103, 104.0]
    days = [day1(100.0),
            Day(np.array([100.0, 99.0]), np.array([100.0, 99.0]), np.array([100.0, 94.0]), np.array([100.0, 96.0]), False),
            day1(96.0), day1(97.0), day1(98.0), day1(99.0)]
    sig = np.array([1, 0, 0, 0, 0, 0.0])
    w = np.array([0.5, 0.9, 0.9, 0.9, 0.9, 0.9])
    p = Params(k=0.05, tick=0.0, comm=C, check="bary")
    r = etap1_run.run_e1("T", days, sig, np.full(6, 1.0), p, w=w)
    assert r.n_orders == 1 and r.n_fills == 1
    tr = r.trades[0]
    assert tr.row0 == 0, "комиссия входа — в строке начала интервала"
    assert r.sides[0] == pytest.approx(0.5), "вход 0.5 в строке 0"
    assert r.sides.sum() == pytest.approx(1.0), "вход + выход, оба с весом 0.5"
    assert r.g[0] == pytest.approx(0.5 * np.log(96.0 / 95.0)), "вход по лимиту 95, выход по close 96, вес 0.5"
    assert not r.g[1:].any()
    assert r.expo[0] == pytest.approx(0.5 * 0.5), "вес × доля баров дня в рынке (1 из 2)"


def test_trade_ended_at_close_of_a_is_outside_window():
    """Сделка, закончившаяся по close a, в окно [a; b) не входит: ни стороны, ни доход (Engine: pos[a] = 0)."""
    px = np.array([100, 101, 102, 103, 104, 105, 106.0])
    sig = np.array([1, 1, 0, 0, 0, 0, 0.0])
    r = etap1_run.run_e1("T", [day1(x) for x in px], sig, np.ones(7), Params(check="bary"), win=(2, 6))
    assert r.sides.sum() == 0 and not r.g.any() and not r.expo.any()


def test_port_rows_equals_port_random_windows():
    rng = np.random.default_rng(5)
    n, npair = 30, 5
    e = Engine.__new__(Engine)
    e.r = {"date": np.tile(np.arange(n), npair)}
    e.first = np.r_[True, np.zeros(n - 1, bool)].repeat(1).tolist() * npair
    e.first = np.array(e.first)
    e.lr_row = rng.normal(0, 0.02, n * npair)
    e.dates = pd.bdate_range("2019-01-02", periods=n)
    e.cnt = np.full(n, float(npair))
    e.kr = np.full(n, 0.07)
    e.rf, e.cal = e.kr / 365.0, np.ones(n)
    pos = rng.integers(-1, 2, n * npair).astype(float)
    a = e.port(pos, C)
    dp = np.diff(pos, prepend=0.0)
    dp[e.first] = pos[e.first]
    last = np.r_[e.first[1:], True]
    b = e.port_rows(pos * e.lr_row, np.abs(dp) + np.where(last, np.abs(pos), 0.0), np.abs(pos), C)
    for col in a.columns:
        assert np.array_equal(a[col].values, b[col].values), col
    bad = e.port_rows(pos * e.lr_row, np.abs(dp), np.abs(pos), C)             # мутация: без платы за выход на конце окна
    assert not np.array_equal(a["r"].values, bad["r"].values)


def test_gate_trades_skips_whole_trade_keeps_open_position():
    pos = np.array([0, 1, 1, 1, 0, -1, -1, 0, 1, 1.0])
    allow = np.array([1, 0, 1, 1, 1, 1, 0, 1, 1, 1], bool)         # вход в день 1 запрещён, в день 5 разрешён
    out = gate_trades(pos, allow)
    assert list(out) == [0, 0, 0, 0, 0, -1, -1, 0, 1, 1], "запрет в день старта снимает сделку целиком; позже — не закрывает"
    flip = gate_trades(np.array([1, 1, -1, -1.0]), np.array([1, 1, 0, 1], bool))
    assert list(flip) == [1, 1, 0, 0], "смена знака — новая сделка, её запрет не закрывает старую"


@pytest.mark.parametrize("name", ["etap1", "etap1_eval", "etap1_data", "etap1_run"])
def test_no_forecast_imports_in_k4_modules(name):             # Ж-047 п. 1 (тест 2.9 для модулей К4)
    from pathlib import Path
    from test_zh047_sigma_rezhim import narusheniya
    assert narusheniya((Path(__file__).resolve().parent.parent / "fastbt" / f"{name}.py").read_text()) == []


def test_mutation_gate_closes_open_trade_caught():
    """Мутация: запрет на каждый день (а не на день старта) — открытая сделка закрывается."""
    pos = np.array([0, 1, 1, 1, 0.0])
    allow = np.array([1, 1, 0, 1, 1], bool)
    assert list(gate_trades(pos, allow)) == [0, 1, 1, 1, 0]
    with pytest.raises(AssertionError):
        assert list(pos * allow) == [0, 1, 1, 1, 0]
