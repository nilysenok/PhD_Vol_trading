"""V6-8: оси К25–К26 (z_f, β до d−1, условия): синтетика с известным ответом + мутации."""
import numpy as np
import pandas as pd
import pytest

from fastbt import vneshnie_uslovie as u

IDX = pd.bdate_range("2019-01-01", periods=300)


def _dx(seed=0):
    return pd.Series(np.random.default_rng(seed).normal(0, 1, len(IDX)), index=IDX)


def test_beta_tochnaya_i_bez_dnya_d():
    dx = _dx()
    r = 3.0 * dx
    r.iloc[-1] = -1000.0 * dx.iloc[-1]                 # аномалия в самом d
    b = u.beta_f(r, dx)
    assert b.iloc[-1] == pytest.approx(3.0)            # день d не входит
    assert b.iloc[:125].isna().all()                   # прогрев: меньше MIN_N пар -> NaN
    assert b.iloc[-2] == pytest.approx(3.0)


def test_mutatsiya_beta_s_dnyom_d_lovitsya(monkeypatch):
    dx = _dx()
    r = 3.0 * dx
    r.iloc[-1] = -1000.0 * dx.iloc[-1]
    monkeypatch.setattr(u, "BETA_LAG", 0)
    assert abs(u.beta_f(r, dx).iloc[-1] - 3.0) > 1.0   # мутация меняет ответ -> тест выше её ловит


def test_beta_okno_252_posledniye_pary():
    dx = _dx(1)
    r = 2.0 * dx
    r.iloc[:40] = -5.0 * dx.iloc[:40]                  # старше окна на d=299: 299-252=47 > 40
    assert u.beta_f(r, dx).iloc[-1] == pytest.approx(2.0)


def test_z_delit_na_vcherashnyuyu_sigma_i_mutaciya(monkeypatch):
    dx = pd.Series(2.0, index=IDX)
    sig = pd.Series(1.0, index=IDX)
    sig.iloc[-1] = 100.0                               # всплеск в день d
    assert u.z_f(dx, sig).iloc[-1] == pytest.approx(2.0)
    assert np.isnan(u.z_f(dx, sig).iloc[0])
    monkeypatch.setattr(u, "SIGMA_LAG", 0)
    assert u.z_f(dx, sig).iloc[-1] == pytest.approx(0.02)


def _zb(z, b):
    i = pd.RangeIndex(len(z))
    return pd.Series(z, index=i, dtype=float), pd.Series(b, index=i, dtype=float)


def test_spokojno():
    z, b = _zb([0.5, -1.0, 1.5, -2.0, 2.5, np.nan], [1] * 6)
    assert u.uslovie("spokojno_1", z, b, 1).tolist() == [True, True, False, False, False, True]
    assert u.uslovie("spokojno_2", z, b, 1).tolist() == [True, True, True, True, False, True]


def test_po_bete_i_protiv_bety_znak():
    z, b = _zb([1.0, 1.0, -1.0, -1.0, 0.0, 1.0], [2, -2, 2, -2, 2, np.nan])
    assert u.uslovie("po_bete", z, b, 1).tolist() == [True, False, False, True, False, True]
    assert u.uslovie("po_bete", z, b, -1).tolist() == [False, True, True, False, False, True]
    assert u.uslovie("protiv_bety", z, b, 1).tolist() == [False, True, True, False, False, True]
    # перепутанный знак стороны: po_bete(side) == protiv_bety(-side) там, где всё известно
    assert (u.uslovie("po_bete", z, b, 1)[:4] == u.uslovie("protiv_bety", z, b, -1)[:4]).all()


def test_net_vsegda_true_i_pustaya_lenta():
    z, b = _zb([5.0, np.nan, -9.0], [np.nan, 1, 1])
    assert u.uslovie("net", z, b, 1).all()
    e = pd.Series(dtype=float)
    assert len(u.uslovie("po_bete", e, e, 1)) == 0
    with pytest.raises(ValueError):
        u.uslovie("xxx", z, b, 1)


def test_nulevaya_dispersiya_dx_beta_nan():
    dx = pd.Series(0.0, index=IDX)
    assert u.beta_f(_dx(2), dx).isna().all()
