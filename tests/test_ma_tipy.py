"""Тесты fastbt.ma_tipy: известные ответы, сверка с источником 6a7a4636, мутации, numba = numpy."""

from math import sqrt

import numpy as np
import pytest

from fastbt import ma_tipy as M

RNG = np.random.default_rng(7)
X = 100 + np.cumsum(RNG.normal(size=300))


# --- эталон: формулы источника (6a7a4636, daily_upper_bound.py, строки 46–82), как есть ---
def src_ema(arr, w):
    alpha = 2.0 / (w + 1)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def src_wma(arr, w):
    weights = np.arange(1, w + 1, dtype=np.float64)
    out = np.full_like(arr, np.nan)
    for i in range(w - 1, len(arr)):
        out[i] = np.dot(arr[i - w + 1:i + 1], weights) / weights.sum()
    return out


def src_hull(arr, w):
    half = max(w // 2, 1)
    diff = 2 * src_wma(arr, half) - src_wma(arr, w)
    return src_wma(diff, max(int(sqrt(w)), 1))


def src_kama(arr, w, fast=2, slow=30):
    fc = 2.0 / (fast + 1)
    sc = 2.0 / (slow + 1)
    out = np.full_like(arr, np.nan)
    out[w - 1] = arr[w - 1]
    for i in range(w, len(arr)):
        direction = abs(arr[i] - arr[i - w])
        volatility = np.sum(np.abs(np.diff(arr[i - w:i + 1])))
        er = 0 if volatility < 1e-12 else direction / volatility
        out[i] = out[i - 1] + (er * (fc - sc) + sc) ** 2 * (arr[i] - out[i - 1])
    return out


def eq(a, b, tol=1e-12):
    assert a.shape == b.shape
    assert np.array_equal(np.isnan(a), np.isnan(b))
    m = ~np.isnan(a)
    assert np.max(np.abs(a[m] - b[m]), initial=0) <= tol


# --- известные ответы ---
@pytest.mark.parametrize("tip", M.TIPY)
def test_konstanta(tip):
    out = M.ma(np.full(60, 5.0), tip, 10)
    assert np.allclose(out[~np.isnan(out)], 5.0, atol=1e-12)
    assert (~np.isnan(out)).sum() > 0


def test_sma_linejnyj_otstaet():
    n = 9
    out = M.sma(np.arange(50.0), n)
    assert np.allclose(out[n - 1:], np.arange(n - 1, 50) - (n - 1) / 2)
    assert np.isnan(out[:n - 1]).all()


def test_ema_alpha():
    out = M.ema(np.array([1.0, 2.0, 4.0]), 3)  # α = 0.5
    assert out.tolist() == [1.0, 1.5, 2.75]


def test_wma_vruchnuyu():
    out = M.wma(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    assert np.isnan(out[:2]).all()
    assert np.allclose(out[2:], [(1 + 4 + 9) / 6, (2 + 6 + 12) / 6, (3 + 8 + 15) / 6])


def test_hma_formula():
    n = 16
    want = M.wma(2 * M.wma(X, n // 2) - M.wma(X, n), 4)
    eq(M.ma(X, "hma", n), want)
    eq(M.ma(X, "hull", n), want)  # имя из источника


def test_dema_tema_formuly():
    e1 = M.ema(X, 10)
    e2 = M.ema(e1, 10)
    e3 = M.ema(e2, 10)
    eq(M.ma(X, "dema", 10), 2 * e1 - e2)
    eq(M.ma(X, "tema", 10), 3 * e1 - 3 * e2 + e3)


def test_kama_kaufman_vruchnuyu():
    # x = 0,1,2,3 (n=3): путь = направление => ER=1, SC=fc²=(2/3)²; out[2]=2, out[3]=2+SC·(3−2)
    out = M.kama(np.array([0.0, 1.0, 2.0, 3.0]), 3)
    assert np.isnan(out[:2]).all() and out[2] == 2.0
    assert out[3] == pytest.approx(2 + (2 / 3) ** 2)
    # плоский ряд: путь 0 => ER=0, SC=sc²=(2/31)², значение не меняется
    assert M.kama(np.full(6, 3.0), 3)[-1] == 3.0
    # ER=0 на ручном примере: 0,1,0 (n=2): d=0, v=2 => ER=0
    o = M.kama(np.array([0.0, 1.0, 0.0]), 2)
    assert o[2] == pytest.approx(1.0 + (2 / 31) ** 2 * (0.0 - 1.0))


def test_kama_parametry_i_oshibki():
    eq(M.ma(X, "kama", 10, fast=3, slow=20), src_kama(X, 10, 3, 20))
    with pytest.raises(ValueError):
        M.ma(X, "nope", 10)
    with pytest.raises(ValueError):
        M.ma(X, "ema", 10, fast=2)
    assert np.isnan(M.kama(np.arange(3.0), 5)).all()


# --- сверка с источником ---
@pytest.mark.parametrize("n", [2, 5, 10, 37])
def test_sverka_s_istochnikom(n):
    eq(M.ma(X, "ema", n), src_ema(X, n), 0)
    eq(M.ma(X, "wma", n), src_wma(X, n), 0)
    eq(M.ma(X, "hma", n), src_hull(X, n), 0)
    eq(M.ma(X, "dema", n), 2 * src_ema(X, n) - src_ema(src_ema(X, n), n), 0)
    eq(M.ma(X, "kama", n), src_kama(X, n), 0)


# --- без заглядывания вперёд; мутации ---
def bez_zaglyadyvaniya(f, d=200):
    """Значения до d включительно не меняются при порче x[d+1:]."""
    y = X.copy()
    y[d + 1:] += 1000.0
    a, b = f(X), f(y)
    return np.array_equal(a[:d + 1], b[:d + 1], equal_nan=True)


@pytest.mark.parametrize("tip", M.TIPY)
def test_net_zaglyadyvaniya(tip):
    assert bez_zaglyadyvaniya(lambda x: M.ma(x, tip, 12))
    assert bez_zaglyadyvaniya(lambda x: M.ma(x, tip, 12, numba=True))


@pytest.mark.parametrize("tip", M.TIPY)
def test_mutaciya_sdvig_plyus_odin_lovitsya(tip):
    def mut(x):  # используется x[d+1]
        return np.roll(M.ma(x, tip, 12), -1)

    assert not bez_zaglyadyvaniya(mut)


def test_mutaciya_neverny_alpha_ema_lovitsya():
    def ema_bad(x, n):
        a = 2.0 / n  # вместо 2/(n+1)
        out = x.copy()
        for i in range(1, len(x)):
            out[i] = a * x[i] + (1 - a) * out[i - 1]
        return out

    assert np.abs(ema_bad(X, 10) - src_ema(X, 10)).max() > 1e-3
    with pytest.raises(AssertionError):
        eq(ema_bad(X, 10), M.ema(X, 10))


# --- быстрый слой = медленный ---
@pytest.mark.parametrize("tip", M.TIPY)
@pytest.mark.parametrize("n", [3, 10, 50])
def test_numba_ravno_numpy(tip, n):
    eq(M.ma(X, tip, n, numba=True), M.ma(X, tip, n), 1e-12)
    if tip == "kama":
        eq(M.ma(X, tip, n, numba=True, fast=3, slow=40), M.ma(X, tip, n, fast=3, slow=40))
