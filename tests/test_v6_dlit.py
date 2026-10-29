"""Лист V6-2: ось dlit (v6_obertka). Известный ответ на синтетике, k=1 = v5 побайтно, причинность, мутации.

Мутации (подкладываются monkeypatch, проверочная функция обязана упасть):
счёт по состоянию сигнала вместо условия; окно с днём d+1; k=1 не равно v5 (>= k → > k).
"""
import numpy as np
import pytest

from fastbt.indicators import compute_base
from fastbt.signals import v5_s12, v5_s34, v5_s56
from fastbt.signals import v6_obertka as W
from fastbt.signals.interface import TickerData

T = 1500
RNG = np.random.default_rng(7)
_t = np.arange(T)
C = 100 * np.exp(np.cumsum(RNG.normal(0, 0.02, T)) + 0.4 * np.sin(_t / 35))
H = C * (1 + RNG.uniform(0, 0.015, T))
L = C * (1 - RNG.uniform(0, 0.015, T))
O = np.r_[C[0], C[:-1]]
V = RNG.uniform(1e3, 5e3, T)

PT = {
    "S1": dict(ma_tip="sma", dev_tip="std", ma_window=20, z_entry=1.0, z_exit="net"),
    "S2": dict(ma_tip="sma", bb_window=20, bb_std=1.5, podtv="none"),
    "S3": dict(okno=10, podtv="none", vhod_tip="podtverzhdenie"),
    "S4": dict(period=10, mnozh=1.0, vhod_tip="podtverzhdenie"),
    "S5": dict(pivot_type="classic", pivot_period="den", uroven=1, atr_filtr=0, filter_set="net"),
    "S6": dict(vwap_window=10, dev_mult=0.5, dir_bias="OFF", filter_set="net"),
}
GEN = {"S1": v5_s12.generate_s1, "S2": v5_s12.generate_s2, "S3": v5_s34.generate_s3_v5,
       "S4": v5_s34.generate_s4_v5, "S5": v5_s56.generate5, "S6": v5_s56.generate6}


def mk(n=T):
    c, h, lo, o, v = C[:n], H[:n], L[:n], O[:n], V[:n]
    lr = np.r_[np.log(c[1:] / c[:-1]), 0.0]
    td = TickerData("X", np.arange(n).astype("datetime64[D]").astype("datetime64[ns]"), o, h, lo, c, v, lr)
    return td, compute_base(c, h, lo, o, v)


def sig_v5(sid, td, ind):
    return GEN[sid](td, ind, [PT[sid]]).sig[0]


def _k1_ravno_v5(sid):
    td, ind = mk()
    s = sig_v5(sid, td, ind)
    assert (s != 0).sum() > 10
    out = W.primenit_dlit(sid, td, ind, PT[sid], s, 1)
    assert out.dtype == s.dtype and out.tobytes() == s.tobytes()


def _monotonno_i_ne_pusto(sid):
    td, ind = mk()
    s = sig_v5(sid, td, ind)
    o2, o3 = (W.primenit_dlit(sid, td, ind, PT[sid], s, k) for k in (2, 3))
    assert np.all((o3 != 0) <= (o2 != 0)) and np.all((o2 != 0) <= (s != 0))
    assert np.all(o2[o2 != 0] == s[o2 != 0])
    assert 0 < (o2 != 0).sum() < (s != 0).sum()


@pytest.mark.parametrize("sid", ["S1", "S2", "S3", "S5", "S6"])
def test_k1_ravno_v5_pobaytno(sid):
    _k1_ravno_v5(sid)


@pytest.mark.parametrize("sid", ["S1", "S2", "S3", "S5", "S6"])
def test_k2_k3_podmnozhestvo_i_ne_pusto(sid):
    _monotonno_i_ne_pusto(sid)


def test_s4_k1_ravno_peresecheniyu_v5():
    _k1_ravno_v5("S4")


def test_s4_k2_sobytie_cherez_den_posle_peresecheniya():
    td, ind = mk()
    s1 = W.primenit_dlit("S4", td, ind, PT["S4"], sig_v5("S4", td, ind), 1)
    s2 = W.primenit_dlit("S4", td, ind, PT["S4"], s1, 2)
    cl, cs = W.usloviya("S4", td, ind, PT["S4"])
    assert (s2 != 0).sum() > 0
    for d in np.flatnonzero(s2):
        cond = cl if s2[d] == 1 else cs
        assert cond[d] and cond[d - 1] and not cond[d - 2]   # пересечение в d-1, по эту сторону и в d
        assert td.n > d and ind["adx14"][d] > 20 and s1[d] in (0, s2[d])


def test_trend_stop_zayavka_svertyvaetsya_v_kanon():
    td, ind = mk()
    for sid in W.TREND:
        s = sig_v5(sid, td, ind)
        p = {k: v for k, v in PT[sid].items() if k != "vhod_tip"}
        for k in (2, 3):
            assert W.primenit_dlit(sid, td, ind, p, s, k).tobytes() == s.tobytes()
            assert W.primenit_dlit(sid, td, ind, dict(p, vhod_tip="stop_zayavka"), s, k).tobytes() == s.tobytes()


def test_dlina_serii_i_maska_izvestnyy_otvet():
    cl = np.array([0, 1, 1, 1, 0, 1, 1, 0, 0, 1], bool)
    cs = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1, 0], bool)
    assert W.dlina_serii(cl).tolist() == [0, 1, 2, 3, 0, 1, 2, 0, 0, 1]
    ok = {k: W.maska_dlit(cl, cs, k) for k in (1, 2, 3)}
    assert np.array_equal(ok[1][0], cl) and np.array_equal(ok[1][1], cs)
    assert np.flatnonzero(ok[2][0]).tolist() == [2, 3, 6] and np.flatnonzero(ok[2][1]).tolist() == [8]
    assert np.flatnonzero(ok[3][0]).tolist() == [3] and not ok[3][1].any()
    with pytest.raises(ValueError):
        W.maska_dlit(cl, cs, 4)


def test_smena_storony_obnulyaet_seriyu():
    cl = np.array([1, 1, 0, 0], bool)
    cs = np.array([0, 0, 1, 1], bool)
    ol, os_ = W.maska_dlit(cl, cs, 2)
    assert ol.tolist() == [False, True, False, False] and os_.tolist() == [False, False, False, True]


def test_s1_izvestnyy_otvet_na_postroennom_ryade():
    n = 400
    c = np.full(n, 100.0)
    c[300:303] = 80.0                                   # три дня ниже полосы подряд (резкий провал)
    td = TickerData("X", np.arange(n).astype("datetime64[D]").astype("datetime64[ns]"),
                    c, c, c, c, np.full(n, 1e3), np.zeros(n))
    ind = compute_base(c, c, c, c, np.full(n, 1e3))
    p = dict(PT["S1"], z_entry=1.0)
    cl, cs = W.usloviya("S1", td, ind, p)
    assert np.flatnonzero(cl).tolist()[:3] == [300, 301, 302] and not cs.any()
    ok2, _ = W.maska_dlit(cl, cs, 2)
    assert ok2[300] == False and ok2[301]   # noqa: E712  в первый день провала при dlit=2 входа нет


def test_prichinnost_prefiks():
    sid = "S1"
    td, ind = mk()
    full = W.primenit_dlit(sid, td, ind, PT[sid], sig_v5(sid, td, ind), 3)
    n = 900
    tdn, indn = mk(n)
    pref = W.primenit_dlit(sid, tdn, indn, PT[sid], sig_v5(sid, tdn, indn), 3)
    assert np.array_equal(full[:n], pref)


# ---- мутации: проверочная функция обязана упасть --------------------------------------------------

def _s3_nezavisimo(k):
    """S3 с podtv=volume: ожидание считается в тесте pandas'ом, независимо от модуля."""
    import pandas as pd
    p = dict(PT["S3"], podtv="volume")
    td, ind = mk()
    s = v5_s34.generate_s3_v5(td, ind, [p]).sig[0]
    ph = pd.Series(H).rolling(10).max().shift(1).to_numpy()
    pl = pd.Series(L).rolling(10).min().shift(1).to_numpy()
    cl, cs = C > ph, C < pl
    exp = s.copy()
    for side, cond in ((1, cl), (-1, cs)):
        run = np.zeros(T, int)
        for t in range(T):
            run[t] = run[t - 1] + 1 if (cond[t] and t) else int(cond[t])
        exp[(s == side) & (run < k)] = 0
    got = W.primenit_dlit("S3", td, ind, p, s, k)
    assert ((s != 0).sum() > (exp != 0).sum() or k == 1) and (exp != 0).sum() > 0
    assert np.array_equal(got, exp)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_s3_nezavisimyy_raschet(k):
    _s3_nezavisimo(k)


def test_mutaciya_schet_po_sostoyaniyu_lovitsya(monkeypatch):
    """Серия считается по сигналу (состоянию, уже прошедшему подтверждение), а не по условию."""
    td, ind = mk()
    s = v5_s34.generate_s3_v5(td, ind, [dict(PT["S3"], podtv="volume")]).sig[0]
    monkeypatch.setattr(W, "usloviya", lambda sid, td, ind, p, cache=None: (s == 1, s == -1))
    with pytest.raises(AssertionError):
        _s3_nezavisimo(2)


def test_mutaciya_okno_s_dnem_d_plus_1_lovitsya(monkeypatch):
    real = W.dlina_serii
    monkeypatch.setattr(W, "dlina_serii", lambda cond, start=None: np.r_[real(cond, start)[1:], 0])
    with pytest.raises(AssertionError):
        _k1_ravno_v5("S1")


def test_mutaciya_k1_ne_ravno_v5_lovitsya(monkeypatch):
    real = W.dlina_serii
    monkeypatch.setattr(W, "dlina_serii", lambda cond, start=None: real(cond, start) - 1)
    with pytest.raises(AssertionError):
        _k1_ravno_v5("S6")
