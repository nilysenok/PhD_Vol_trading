"""Независимые тесты V6-8 (К25–К26): z_f, β до d−1, условия. Тестировщик, не автор."""
import numpy as np
import pandas as pd
import pytest

from fastbt import vneshnie_uslovie as V


def _idx(n):
    return pd.bdate_range("2019-01-01", periods=n)


def _series(n, seed):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 1, n), index=_idx(n))


def _beta_loop(r, dx, d, window, min_n):
    """Эталон циклом: МНК по парам из окна [d-window, d-1]."""
    lo = max(0, d - window)
    xs, ys = dx.iloc[lo:d].to_numpy(), r.iloc[lo:d].to_numpy()
    m = ~(np.isnan(xs) | np.isnan(ys))
    if m.sum() < min_n:
        return np.nan
    xs, ys = xs[m], ys[m]
    vx = xs.var(ddof=1)
    if vx <= 0:
        return np.nan
    return np.cov(xs, ys, ddof=1)[0, 1] / vx


def test_z_known_answer():
    ix = _idx(4)
    dx = pd.Series([1.0, 2.0, 3.0, 4.0], index=ix)
    sg = pd.Series([10.0, 20.0, 30.0, 40.0], index=ix)
    z = V.z_f(dx, sg)
    assert np.isnan(z.iloc[0])
    assert z.iloc[1] == pytest.approx(2.0 / 10.0)
    assert z.iloc[3] == pytest.approx(4.0 / 30.0)


def test_z_causal_sigma_today_ignored():
    dx, sg = _series(50, 1), _series(50, 2).abs() + 0.5
    base = V.z_f(dx, sg)
    sg2 = sg.copy()
    sg2.iloc[30:] *= 100  # порча σ с дня 30
    z2 = V.z_f(dx, sg2)
    # σ дня 30 влияет только начиная с z дня 31
    pd.testing.assert_series_equal(base.iloc[:31], z2.iloc[:31])
    assert z2.iloc[31] != base.iloc[31]


def test_beta_matches_loop_and_window_excludes_d():
    n, w, mn = 80, 20, 10
    r, dx = _series(n, 3), _series(n, 4)
    b = V.beta_f(r, dx, window=w, min_n=mn)
    for d in (5, 9, 10, 25, 79):
        exp = _beta_loop(r, dx, d, w, mn)
        if np.isnan(exp):
            assert np.isnan(b.iloc[d]), d
        else:
            assert b.iloc[d] == pytest.approx(exp, rel=1e-9), d


def test_beta_known_slope_exact():
    n = 60
    dx = _series(n, 5)
    r = 2.5 * dx
    b = V.beta_f(r, dx, window=30, min_n=10)
    assert b.iloc[40] == pytest.approx(2.5)
    assert b.iloc[:10].isna().all()  # прогрев: d-1 < min_n пар


def test_beta_causality_future_and_day_d():
    n, K = 70, 40
    r, dx = _series(n, 6), _series(n, 7)
    b = V.beta_f(r, dx, window=20, min_n=10)
    r2, dx2 = r.copy(), dx.copy()
    r2.iloc[K:] = 1e6
    dx2.iloc[K:] = -1e6 * np.arange(1, n - K + 1)
    b2 = V.beta_f(r2, dx2, window=20, min_n=10)
    # значения до и включая K не меняются: β_K использует до K-1
    pd.testing.assert_series_equal(b.iloc[:K + 1], b2.iloc[:K + 1])
    assert b.iloc[K + 1] != b2.iloc[K + 1]


def test_beta_nan_handling_and_short_series():
    ix = _idx(5)
    r = pd.Series([1.0, 2, np.nan, 4, 5], index=ix)
    dx = pd.Series([1.0, 2, 3, np.nan, 5], index=ix)
    b = V.beta_f(r, dx)  # ряд короче min_n
    assert b.isna().all()
    e = V.beta_f(pd.Series(dtype=float), pd.Series(dtype=float))
    assert len(e) == 0
    # константный dx: var = 0 -> NaN, а не inf
    c = V.beta_f(_series(300, 8), pd.Series(1.0, index=_idx(300)))
    assert c.isna().all()


def test_net_all_true_even_with_nan():
    ix = _idx(6)
    z = pd.Series([np.nan, 5, -5, 0, 1, 2.0], index=ix)
    b = pd.Series(np.nan, index=ix)
    out = V.uslovie("net", z, b, 1)
    assert out.all() and len(out) == 6


@pytest.mark.parametrize("name,lim", [("spokojno_1", 1.0), ("spokojno_2", 2.0)])
def test_spokojno_boundary_inclusive(name, lim):
    ix = _idx(6)
    z = pd.Series([lim, -lim, lim + 1e-9, -lim - 1e-9, np.nan, 0.0], index=ix)
    b = pd.Series(np.nan, index=ix)  # β не нужна для spokojno
    out = V.uslovie(name, z, b, 1)
    assert list(out) == [True, True, False, False, True, True]


def test_po_bete_protiv_bety_truth_table():
    ix = _idx(8)
    z = pd.Series([1.0, 1.0, -1.0, -1.0, 0.0, np.nan, 1.0, 1.0], index=ix)
    b = pd.Series([2.0, -2.0, 2.0, -2.0, 3.0, 3.0, np.nan, 0.0], index=ix)
    for side in (1, -1):
        po = V.uslovie("po_bete", z, b, side)
        pr = V.uslovie("protiv_bety", z, b, side)
        sg = [1, -1, -1, 1, 0]  # sign(b*z) по первым пяти
        for i, s in enumerate(sg):
            assert po.iloc[i] == (s == side), (side, i)
            assert pr.iloc[i] == (s == -side), (side, i)
        assert po.iloc[5] and pr.iloc[5] and po.iloc[6] and pr.iloc[6]  # неизвестное -> True
        assert not po.iloc[7] and not pr.iloc[7]  # β=0 известна: sign 0 не пускает


def test_side_as_series():
    ix = _idx(3)
    z = pd.Series([1.0, 1.0, 1.0], index=ix)
    b = pd.Series([1.0, 1.0, 1.0], index=ix)
    side = pd.Series([1, -1, 1], index=ix)
    assert list(V.uslovie("po_bete", z, b, side)) == [True, False, True]


def test_unknown_name_raises_and_empty():
    with pytest.raises(ValueError):
        V.uslovie("xxx", pd.Series(dtype=float), pd.Series(dtype=float), 1)
    e = pd.Series(dtype=float)
    assert len(V.uslovie("po_bete", e, e, 1)) == 0


def test_off_equals_v5_mask():
    """«выкл.» = все True: маска входа v5 не меняется при умножении на net."""
    ix = _idx(30)
    v5 = pd.Series(np.random.default_rng(9).random(30) > 0.5, index=ix)
    z, b = _series(30, 1), _series(30, 2)
    assert ((v5 & V.uslovie("net", z, b, 1)) == v5).all()
