"""Независимые тесты тестировщика к V6-5 (ось norm: okno | s252 в S1, S2, S4, S6).

Свои данные; эталоны s252 считаются в тесте pandas'ом / циклом. Проверяется: norm=okno и отсутствие ключа == v5
побайтно (sig, выходы, сила), прогрев 251 день - вход невозможен (NaN, не 1e-12), причинность, неподвижность оси
period в S4 при s252, неизвестное значение - ValueError, граница дня 251 / 252, влияние на кэш при смешанной сетке.
"""
import numpy as np
import pandas as pd
import pytest

from fastbt.indicators import compute_base
from fastbt.signals import v5_s12, v5_s34, v5_s56
from fastbt.signals.interface import WARMUP, TickerData
from fastbt.signals.s34 import supertrend
from fastbt.signals.s56_kern import _vwap

T = 1100
_R = np.random.default_rng(777)
_t = np.arange(T)
C0 = 40 * np.exp(np.cumsum(_R.normal(0, 0.019, T)) + 0.5 * np.sin(_t / 33) + 0.25 * np.cos(_t / 9))
H0 = C0 * (1 + _R.uniform(0.001, 0.017, T))
L0 = C0 * (1 - _R.uniform(0.001, 0.017, T))
O0 = np.r_[C0[0], C0[:-1]]
V0 = _R.uniform(1e3, 7e3, T)


def mk(c=C0, h=H0, lo=L0, o=O0, v=V0):
    lr = np.r_[np.log(c[1:] / c[:-1]), 0.0]
    td = TickerData("N", np.arange(len(c)).astype("datetime64[D]").astype("datetime64[ns]"), o, h, lo, c, v, lr)
    return td, compute_base(c, h, lo, o, v)


P1 = dict(ma_tip="sma", dev_tip="std", ma_window=20, z_entry=0.8, z_exit="net")
P1A = dict(P1, dev_tip="atr")
P2 = dict(ma_tip="sma", bb_window=20, bb_std=1.0, podtv="none")
P4 = dict(period=10, mnozh=1.5)
P6 = dict(vwap_window=15, dev_mult=0.5, dir_bias="OFF", filter_set="net")
CASES = [("S1", v5_s12.generate_s1, P1), ("S1a", v5_s12.generate_s1, P1A), ("S2", v5_s12.generate_s2, P2),
         ("S4", v5_s34.generate_s4_v5, P4), ("S6", v5_s56.generate6, P6)]
IDS = [c[0] for c in CASES]


def gen(fn, p, **kw):
    td, ind = mk(**kw)
    return fn(td, ind, [p])


# ---------------- okno == v5 побайтно ----------------
@pytest.mark.parametrize("name,fn,p", CASES, ids=IDS)
def test_okno_and_missing_key_identical_all_arrays(name, fn, p):
    a = gen(fn, p)
    b = gen(fn, dict(p, norm="okno"))
    assert (a.sig != 0).sum() > 5
    for x, y in ((a.sig, b.sig), (a.exit_l, b.exit_l), (a.exit_s, b.exit_s), (a.strength, b.strength)):
        assert x.dtype == y.dtype and x.tobytes() == y.tobytes()


@pytest.mark.parametrize("name,fn,p", CASES, ids=IDS)
def test_s252_changes_signal(name, fn, p):
    a, b = gen(fn, p), gen(fn, dict(p, norm="s252"))
    assert (b.sig != 0).sum() > 0, "тест вырожден: s252 без входов"
    assert not np.array_equal(a.sig, b.sig)


@pytest.mark.parametrize("name,fn,p", CASES, ids=IDS)
@pytest.mark.parametrize("bad", ["s250", "S252", "", None, 252])
def test_unknown_norm_raises(name, fn, p, bad):
    td, ind = mk()
    with pytest.raises(ValueError):
        fn(td, ind, [dict(p, norm=bad)])


# ---------------- прогрев: 251 день без входов ----------------
@pytest.mark.parametrize("name,fn,p", CASES, ids=IDS)
def test_no_entries_in_first_251_days(name, fn, p):
    s = gen(fn, dict(p, norm="s252")).sig[0]
    assert WARMUP < 251
    assert not s[:251].any()
    assert (s[251:] != 0).sum() > 0


def test_warmup_boundary_day_251_is_first_possible():
    """Ряд с резким выбросом ровно на дне 250 (нет σ252) и 251 (есть): вход возможен только на 251."""
    n = 400
    r = np.random.default_rng(3)
    c = 100 + np.cumsum(r.normal(0, 0.5, n))
    c[250] = 40.0      # огромный провал (z << -z_entry при любой σ), но σ252 ещё не определена
    c[251] = 40.0
    c[252:] = 100 + np.cumsum(r.normal(0, 0.5, n - 252))
    h, lo = c * 1.001, c * 0.999
    o, v = np.r_[c[0], c[:-1]], np.full(n, 3e3)
    td, ind = mk(c, h, lo, o, v)
    p = dict(P1, norm="s252", z_exit="net")
    s = v5_s12.generate_s1(td, ind, [p]).sig[0]
    assert s[250] == 0 and s[251] == 1
    s_ok = v5_s12.generate_s1(td, ind, [dict(P1)]).sig[0]
    assert s_ok[250] == 1       # в v5 (okno) вход на 250 есть - граница действительно разделяет режимы


# ---------------- эталоны s252 независимо ----------------
def _loop_s1(z, ze, t0):
    out = np.zeros(len(z), np.int8)
    for t in range(len(z)):
        if t < t0 or np.isnan(z[t]):
            continue
        if z[t] < -ze:
            out[t] = 1
        elif z[t] > ze:
            out[t] = -1
    return out


def test_s1_std252_vs_pandas():
    s = pd.Series(C0)
    z = ((s - s.rolling(20).mean()) / s.rolling(252).std(ddof=1)).to_numpy()
    exp = _loop_s1(z, 0.8, WARMUP)
    got = gen(v5_s12.generate_s1, dict(P1, norm="s252")).sig[0]
    assert (exp != 0).sum() > 10
    assert np.array_equal(got, exp)


def test_s1_atr252_no_signal_before_atr_defined_and_uses_atr252():
    from fastbt.indicators import calc_atr
    atr = calc_atr(H0, L0, C0, 252)
    s = pd.Series(C0)
    z = ((s - s.rolling(20).mean()) / atr).to_numpy()
    exp = _loop_s1(z, 0.8, WARMUP)
    got = gen(v5_s12.generate_s1, dict(P1A, norm="s252")).sig[0]
    assert np.array_equal(got, exp)
    first_valid = int(np.argmax(~np.isnan(atr)))
    assert not got[:first_valid].any()


def test_s2_std252_vs_pandas():
    s = pd.Series(C0)
    ma, sd = s.rolling(20).mean().to_numpy(), s.rolling(252).std(ddof=1).to_numpy()
    exp = np.zeros(T, np.int8)
    for t in range(WARMUP, T):
        if np.isnan(ma[t]) or np.isnan(sd[t]):
            continue
        if C0[t] <= ma[t] - 1.0 * sd[t]:
            exp[t] = 1
        elif C0[t] >= ma[t] + 1.0 * sd[t]:
            exp[t] = -1
    got = gen(v5_s12.generate_s2, dict(P2, norm="s252")).sig[0]
    assert (exp != 0).sum() > 10
    assert np.array_equal(got, exp)


def test_s6_dev252_vs_pandas():
    vwap, _ = _vwap(C0, H0, L0, V0, 15)
    d = pd.Series(C0 - vwap).rolling(252).std(ddof=1).to_numpy()
    d = np.where(d < 1e-12, 1e-12, d)           # NaN остаётся NaN
    exp = np.zeros(T, np.int8)
    for t in range(WARMUP, T):
        if np.isnan(vwap[t]) or np.isnan(d[t]):
            continue
        if C0[t] < vwap[t] - 0.5 * d[t]:
            exp[t] = 1
        elif C0[t] > vwap[t] + 0.5 * d[t]:
            exp[t] = -1
    ss = gen(v5_s56.generate6, dict(P6, norm="s252"))
    assert (exp != 0).sum() > 10
    assert np.array_equal(ss.sig[0], exp)
    # dev в extra у s252 - окно (по заявлению автора не менялся) или 252; фиксируем фактическое поведение
    ss0 = gen(v5_s56.generate6, P6)
    assert ss.extra["vwap"].tobytes() == ss0.extra["vwap"].tobytes()


def test_s4_s252_equals_supertrend_252_and_period_axis_frozen():
    td, ind = mk()
    a = v5_s34.generate_s4_v5(td, ind, [dict(P4, norm="s252")]).sig[0]
    b = v5_s34.generate_s4_v5(td, ind, [dict(P4, period=60, norm="s252")]).sig[0]
    c = v5_s34.generate_s4_v5(td, ind, [dict(P4, period=3, norm="s252")]).sig[0]
    assert np.array_equal(a, b) and np.array_equal(a, c) and (a != 0).sum() > 0
    st = supertrend(H0, L0, C0, 252, 1.5)
    assert not a[: int(np.argmax(~np.isnan(st)))].any()
    # mnozh остаётся живой осью
    d = v5_s34.generate_s4_v5(td, ind, [dict(P4, mnozh=3.0, norm="s252")]).sig[0]
    assert not np.array_equal(a, d)


# ---------------- смешанная сетка и кэш ----------------
def test_mixed_grid_equals_separate_runs():
    """Точки okno и s252 в одной сетке (общий cache) дают то же, что по отдельности (нет загрязнения кэша)."""
    td, ind = mk()
    grid1 = [dict(P1), dict(P1, norm="s252"), dict(P1, ma_window=50), dict(P1, ma_window=50, norm="s252"), dict(P1)]
    ss = v5_s12.generate_s1(td, ind, grid1)
    for i, p in enumerate(grid1):
        one = v5_s12.generate_s1(td, ind, [p])
        assert ss.sig[i].tobytes() == one.sig[0].tobytes()
        assert ss.strength[i].tobytes() == one.strength[0].tobytes()
    grid6 = [dict(P6), dict(P6, norm="s252"), dict(P6)]
    ss6 = v5_s56.generate6(td, ind, grid6)
    for i, p in enumerate(grid6):
        one = v5_s56.generate6(td, ind, [p])
        assert ss6.sig[i].tobytes() == one.sig[0].tobytes()


def test_s1_ma_window_252_okno_not_polluted_by_s252():
    """Ключ кэша ("std", 252): okno с окном 252 после s252 в той же сетке обязан быть как v5 (1e-12 в прогреве, не NaN)."""
    td, ind = mk()
    p_okno = dict(P1, ma_window=252)
    alone = v5_s12.generate_s1(td, ind, [p_okno]).sig[0]
    mixed = v5_s12.generate_s1(td, ind, [dict(P1, norm="s252"), p_okno]).sig[1]
    assert alone.tobytes() == mixed.tobytes()


# ---------------- причинность ----------------
@pytest.mark.parametrize("name,fn,p", CASES, ids=IDS)
def test_causality_s252(name, fn, p):
    K = 800
    full = gen(fn, dict(p, norm="s252"))
    r = np.random.default_rng(4)
    c, h, lo, o, v = (a.copy() for a in (C0, H0, L0, O0, V0))
    f = np.exp(r.normal(0, 0.25, T - K))
    for a in (c, h, lo, o):
        a[K:] *= f
    v[K:] *= r.uniform(0.2, 4, T - K)
    bad = gen(fn, dict(p, norm="s252"), c=c, h=h, lo=lo, o=o, v=v)
    assert np.array_equal(full.sig[0][:K], bad.sig[0][:K])
    assert (full.sig[0][:K] != 0).sum() > 0
