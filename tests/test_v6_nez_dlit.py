"""Независимые тесты тестировщика к V6-2 (ось dlit, fastbt/signals/v6_obertka.py).

Свои данные (seed 2024, иной ряд, чем в тестах автора), эталоны считаются циклами и pandas в самом тесте.
Проверяется: k=1 == v5 побайтно (все семьи), причинность (порча данных после K не меняет значения до K),
известный ответ для длины серии / S1 / S2 / S3 / S4, граничные случаи (k вне оси, NaN, вход не мутируется).
"""
import numpy as np
import pandas as pd
import pytest

from fastbt.indicators import compute_base
from fastbt.signals import v5_s12, v5_s34, v5_s56
from fastbt.signals import v6_obertka as W
from fastbt.signals.interface import TickerData
from fastbt.signals.s34 import supertrend

T = 1300
_R = np.random.default_rng(2024)
_t = np.arange(T)
C0 = 50 * np.exp(np.cumsum(_R.normal(0, 0.017, T)) + 0.5 * np.sin(_t / 27) + 0.3 * np.cos(_t / 11))
H0 = C0 * (1 + _R.uniform(0.001, 0.02, T))
L0 = C0 * (1 - _R.uniform(0.001, 0.02, T))
O0 = np.r_[C0[0], C0[:-1]] * (1 + _R.normal(0, 0.003, T))
V0 = _R.uniform(2e3, 9e3, T)

P = {
    "S1": dict(ma_tip="sma", dev_tip="std", ma_window=15, z_entry=0.8, z_exit="net"),
    "S2": dict(ma_tip="sma", bb_window=15, bb_std=1.2, podtv="none"),
    "S3": dict(okno=8, podtv="none", vhod_tip="podtverzhdenie"),
    "S4": dict(period=7, mnozh=1.5, vhod_tip="podtverzhdenie"),
    "S5": dict(pivot_type="classic", pivot_period="den", uroven=1, atr_filtr=0, filter_set="net"),
    "S6": dict(vwap_window=12, dev_mult=0.4, dir_bias="OFF", filter_set="net"),
}
GEN = {"S1": v5_s12.generate_s1, "S2": v5_s12.generate_s2, "S3": v5_s34.generate_s3_v5,
       "S4": v5_s34.generate_s4_v5, "S5": v5_s56.generate5, "S6": v5_s56.generate6}
SIDS = list(P)


def mk(c=None, h=None, lo=None, o=None, v=None, n=T):
    c, h, lo, o, v = [(a if a is not None else b)[:n] for a, b in ((c, C0), (h, H0), (lo, L0), (o, O0), (v, V0))]
    lr = np.r_[np.log(c[1:] / c[:-1]), 0.0]
    dates = np.arange(n).astype("datetime64[D]").astype("datetime64[ns]")
    return TickerData("Y", dates, o, h, lo, c, v, lr), compute_base(c, h, lo, o, v)


def v5(sid, td, ind):
    return GEN[sid](td, ind, [P[sid]]).sig[0]


# ---------------- k = 1 == v5 побайтно ----------------
@pytest.mark.parametrize("sid", SIDS)
def test_k1_byte_equal_v5(sid):
    td, ind = mk()
    s = v5(sid, td, ind)
    assert (s != 0).sum() > 5, "тест вырожден: v5 без входов"
    out = W.primenit_dlit(sid, td, ind, P[sid], s, 1)
    assert out.dtype == s.dtype and out.shape == s.shape and out.tobytes() == s.tobytes()


@pytest.mark.parametrize("sid", SIDS)
def test_input_sig_not_mutated_and_output_is_new(sid):
    td, ind = mk()
    s = v5(sid, td, ind)
    snap = s.copy()
    for k in (1, 2, 3):
        out = W.primenit_dlit(sid, td, ind, P[sid], s, k)
        assert out is not s
        assert s.tobytes() == snap.tobytes()


# ---------------- причинность ----------------
@pytest.mark.parametrize("sid", SIDS)
@pytest.mark.parametrize("k", [2, 3])
def test_causality_future_corruption(sid, k):
    K = 700
    td, ind = mk()
    full = W.primenit_dlit(sid, td, ind, P[sid], v5(sid, td, ind), k)
    r = np.random.default_rng(5)
    c, h, lo, o, v = (a.copy() for a in (C0, H0, L0, O0, V0))
    f = np.exp(r.normal(0, 0.3, T - K))
    for a in (c, h, lo, o):
        a[K:] = a[K:] * f
    v[K:] = v[K:] * r.uniform(0.1, 5, T - K)
    td2, ind2 = mk(c, h, lo, o, v)
    bad = W.primenit_dlit(sid, td2, ind2, P[sid], v5(sid, td2, ind2), k)
    assert np.array_equal(full[:K], bad[:K])
    assert (full[:K] != 0).sum() > 0, "проверка вырождена: нет сигналов до K"


# ---------------- известный ответ: длина серии ----------------
def test_dlina_serii_hand_cases():
    c = np.array([1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1], bool)
    assert W.dlina_serii(c).tolist() == [1, 2, 3, 0, 1, 0, 0, 1, 2, 3, 4]
    assert W.dlina_serii(np.zeros(0, bool)).tolist() == []
    assert W.dlina_serii(np.zeros(5, bool)).tolist() == [0] * 5
    assert W.dlina_serii(np.ones(4, bool)).tolist() == [1, 2, 3, 4]
    # start: серия может начаться только в отмеченный день, но продолжается без отметки
    st = np.array([0, 1, 0, 0, 0, 0, 1], bool)
    cond = np.array([1, 1, 1, 0, 1, 1, 1], bool)
    assert W.dlina_serii(cond, st).tolist() == [0, 1, 2, 0, 0, 0, 1]


def test_maska_dlit_hand_cases_and_boundaries():
    cl = np.array([1, 1, 1, 1, 0, 1], bool)
    cs = np.array([0, 0, 0, 0, 1, 0], bool)
    for k, exp_l in ((1, [0, 1, 2, 3, 5]), (2, [1, 2, 3]), (3, [2, 3])):
        ol, os_ = W.maska_dlit(cl, cs, k)
        assert np.flatnonzero(ol).tolist() == exp_l
        assert np.flatnonzero(os_).tolist() == ([4] if k == 1 else [])
    for bad in (0, 4, -1, 2.5, None):
        with pytest.raises(ValueError):
            W.maska_dlit(cl, cs, bad)


@pytest.mark.parametrize("bad", [0, 4, -1])
def test_primenit_rejects_k_outside_axis(bad):
    td, ind = mk()
    s = v5("S1", td, ind)
    with pytest.raises(ValueError):
        W.primenit_dlit("S1", td, ind, P["S1"], s, bad)


# ---------------- условия против независимого расчёта ----------------
def test_s1_usloviya_vs_pandas():
    td, ind = mk()
    cl, cs = W.usloviya("S1", td, ind, P["S1"])
    s = pd.Series(C0)
    ma = s.rolling(15).mean()
    sd = s.rolling(15).std(ddof=1)     # допущение: dev_tip=std это std(close), ddof=1
    z = np.nan_to_num(((s - ma) / sd).to_numpy(), nan=0)
    exp_l, exp_s = z < -0.8, z > 0.8
    assert (exp_l | exp_s).sum() > 50
    assert np.array_equal(cl[40:], exp_l[40:]) and np.array_equal(cs[40:], exp_s[40:])
    assert not (cl & cs).any()


def test_s2_usloviya_vs_pandas():
    td, ind = mk()
    cl, cs = W.usloviya("S2", td, ind, P["S2"])
    s = pd.Series(C0)
    ma, sd = s.rolling(15).mean(), s.rolling(15).std(ddof=1)
    lo, hi = (ma - 1.2 * sd).to_numpy(), (ma + 1.2 * sd).to_numpy()
    assert np.array_equal(cl[40:], (C0 <= lo)[40:]) and np.array_equal(cs[40:], (C0 >= hi)[40:])
    assert cl.sum() > 20 and cs.sum() > 20


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("sid", ["S1", "S2"])
def test_s12_masked_signal_vs_loop(sid, k):
    td, ind = mk()
    s = v5(sid, td, ind)
    cl, cs = W.usloviya(sid, td, ind, P[sid])
    exp = s.copy()
    for t in range(T):
        for side, cond in ((1, cl), (-1, cs)):
            if s[t] == side and not all(t - j >= 0 and cond[t - j] for j in range(k)):
                exp[t] = 0
    out = W.primenit_dlit(sid, td, ind, P[sid], s, k)
    assert np.array_equal(out, exp)
    assert np.all(out[out != 0] == s[out != 0])
    if k > 1:
        assert 0 < (out != 0).sum() < (s != 0).sum()


def test_s3_usloviya_vs_donchian_loop():
    td, ind = mk()
    cl, cs = W.usloviya("S3", td, ind, P["S3"])
    w = 8
    n_l = n_s = 0
    for t in range(w + 1, T):
        assert cl[t] == (C0[t] > H0[t - w:t].max())     # канал по t-1, текущий бар не входит
        assert cs[t] == (C0[t] < L0[t - w:t].min())
        n_l += cl[t]
        n_s += cs[t]
    assert n_l > 20 and n_s > 20


def test_s4_event_vs_loop():
    td, ind = mk()
    p = P["S4"]
    st = supertrend(H0, L0, C0, p["period"], p["mnozh"])
    adx, vf = ind["adx14"], ind["vf_pass"]
    s1 = v5("S4", td, ind)
    for k in (2, 3):
        exp = np.zeros(T, np.int8)
        for side, cmp_ in ((1, lambda a, b: a > b), (-1, lambda a, b: a < b)):
            run = 0
            for t in range(1, T):
                on = (not np.isnan(st[t])) and cmp_(C0[t], st[t])
                if not on:
                    run = 0
                    continue
                if run > 0:
                    run += 1
                elif not np.isnan(st[t - 1]) and not cmp_(C0[t - 1], st[t - 1]):
                    run = 1
                if run == k and t >= 200 and adx[t] > 20 and vf[t] and exp[t] == 0:
                    exp[t] = side
        got = W.primenit_dlit("S4", td, ind, p, s1, k)
        assert (exp != 0).sum() > 0
        assert np.array_equal(got, exp), (k, np.flatnonzero(got != exp)[:5])


def test_s4_k2_independent_of_input_sig():
    """Для S4 при k>=2 выход считается из данных; sig=нули результат не меняет."""
    td, ind = mk()
    s = v5("S4", td, ind)
    a = W.primenit_dlit("S4", td, ind, P["S4"], s, 2)
    b = W.primenit_dlit("S4", td, ind, P["S4"], np.zeros_like(s), 2)
    assert np.array_equal(a, b)


# ---------------- монотонность по k, свёртка трендов ----------------
@pytest.mark.parametrize("sid", ["S1", "S2", "S3", "S5", "S6"])
def test_monotone_in_k_and_signs_preserved(sid):
    td, ind = mk()
    s = v5(sid, td, ind)
    o = {k: W.primenit_dlit(sid, td, ind, P[sid], s, k) for k in (1, 2, 3)}
    assert np.all((o[3] != 0) <= (o[2] != 0)) and np.all((o[2] != 0) <= (o[1] != 0))
    for k in (2, 3):
        assert np.all(o[k][o[k] != 0] == s[o[k] != 0])


@pytest.mark.parametrize("sid", ["S3", "S4"])
def test_trend_without_vhod_tip_is_canon(sid):
    td, ind = mk()
    p = {k: v for k, v in P[sid].items() if k != "vhod_tip"}
    s = GEN[sid](td, ind, [p]).sig[0]
    for k in (1, 2, 3):
        out = W.primenit_dlit(sid, td, ind, p, s, k)
        assert out.tobytes() == s.tobytes() and out is not s


# ---------------- граничные случаи ----------------
def _flat(n=400, dip=None):
    c = np.full(n, 100.0)
    if dip:
        c[dip[0]:dip[1]] = 70.0
    td = TickerData("Z", np.arange(n).astype("datetime64[D]").astype("datetime64[ns]"), c, c, c, c,
                    np.full(n, 1e3), np.zeros(n))
    return td, compute_base(c, c, c, c, np.full(n, 1e3))


def test_constant_series_gives_no_entries():
    td, ind = _flat(300)
    for sid in ("S1", "S3"):
        cl, cs = W.usloviya(sid, td, ind, P[sid])
        assert not cl.any() and not cs.any()
        assert not W.primenit_dlit(sid, td, ind, P[sid], np.zeros(300, np.int8), 3).any()


def test_s1_two_day_dip_k3_empty_k2_single():
    td, ind = _flat(400, (300, 302))
    cl, cs = W.usloviya("S1", td, ind, P["S1"])
    assert cl[300] and cl[301]
    ol2, _ = W.maska_dlit(cl, cs, 2)
    ol3, _ = W.maska_dlit(cl, cs, 3)
    assert ol2[300:302].tolist() == [False, True]
    assert not ol3[300:305].any()
