"""CE-3: к2 при строгом исполнении. Синтетика, без данных. Мутации: лаг 0 ≠ исходному; сдвиг через границу пары;
ноги сдвинуты вместе с g; лаги не те, что у старого к2."""
import numpy as np
import pandas as pd
import pytest

from fastbt import fast_kontrol as fk
from fastbt import fast_kontrol_strogiy as ks
from fastbt import strogoe
from fastbt.s9 import Engine

C = 0.0004
NP, T = 2, 60                                    # две пары, один тикер на пару


def eng():
    rng = np.random.default_rng(5)
    n = NP * T
    e = Engine.__new__(Engine)
    e.r = {"date": np.tile(np.arange(T), NP), "tick": np.repeat(np.arange(NP), T), "bar": np.tile(np.arange(T), NP)}
    e.first = np.zeros(n, bool)
    e.first[::T] = True
    e.dates = pd.bdate_range("2019-01-02", periods=T)
    e.cnt, e.kr = np.full(T, float(NP)), np.full(T, 0.07)
    e.rf, e.cal = np.full(T, 0.07 / 365), np.ones(T)
    e.lr_row = rng.normal(0, 0.01, n)
    nr, dr = rng.normal(0, 0.003, n), rng.normal(0, 0.01, n)
    nr[T - 1::T] = dr[T - 1::T] = 0.0
    return e, (nr, dr)


def data():
    rng = np.random.default_rng(6)
    n = NP * T
    return rng.choice([-1.0, 0.0, 1.0], n), (rng.random(n) > 0.4).astype(float)


def test_lag0_raven_iskhodnoy():                 # mutations № 9
    e, nog = eng()
    A, g = data()
    a = ks.shift_strogiy(e, A, g, 0, C, nog)
    b = strogoe.port_strogiy(e, A * g, C, nogi=nog)
    for col in a.columns:
        assert np.array_equal(a[col].values, b[col].values), col
    c = ks.shift_strogiy(e, A, g, 25, C, nog)
    assert not np.allclose(c["gross"].values, a["gross"].values)


def test_5_tochek_ravny_port_strogiy():          # эталон — явный цикл по парам
    e, nog = eng()
    A, g = data()
    for lag in (21, 25, 30, 37, 39):
        gs = np.empty_like(g)
        for p in range(NP):
            for j in range(T):
                gs[p * T + j] = g[p * T + (j - lag) % T]
        ref = strogoe.port_strogiy(e, A * gs, C, nogi=nog)
        got = ks.shift_strogiy(e, A, g, np.full(NP, lag), C, nog)
        assert np.abs(got["excess"].values - ref["excess"].values).max() <= 1e-12
        assert np.abs(got["r"].values - ref["r"].values).max() <= 1e-12


def test_sdvig_ne_peresekaet_paru():
    e, _ = eng()
    g = np.r_[np.zeros(T), np.ones(T)]
    assert ks.shift_g(e, g, 30)[:T].sum() == 0 and ks.shift_g(e, g, 30)[T:].sum() == T


def test_nogi_ne_sdvigayutsya():                 # мутация «ноги сдвинуты» дала бы другой результат
    e, (nr, dr) = eng()
    A, g = data()
    got = ks.shift_strogiy(e, A, g, 30, C, (nr, dr))
    moved = strogoe.port_strogiy(e, A * ks.shift_g(e, g, 30), C, nogi=(ks.shift_g(e, nr, 30), ks.shift_g(e, dr, 30)))
    assert not np.allclose(got["gross"].values, moved["gross"].values)


def test_lagi_te_zhe_chto_u_starogo_k2():
    e, _ = eng()
    L = ks.lags_k2(e, n_shift=50)
    assert np.array_equal(L, fk.draw_lags(np.array([T, T]), 50, fk.SEED, fk.MIN_LAG))
    assert L.min() >= fk.MIN_LAG and L.shape == (50, NP)


def test_sharpe_k2_i_p():
    e, nog = eng()
    A, g = data()
    L = ks.lags_k2(e, n_shift=20)
    sh = ks.k2_sharpes_strogiy(e, A, g, C, nog, L)
    ref = [ks.sharpe(ks.shift_strogiy(e, A, g, L[b], C, nog)["excess"].values) for b in range(20)]
    assert np.array_equal(sh, ref) and np.isfinite(sh).all()
    assert fk.p_perm(sh.max() + 1, sh) == pytest.approx(1 / 21)
    assert ks.sharpe(np.zeros(5)) == 0.0
