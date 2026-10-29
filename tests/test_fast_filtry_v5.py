"""Лист 4 из П-0: numba-версии calc_adx, maska_atr_expand, rangi_v6.rang = эталон побайтно (допуск 0).

Данные: синтетика и дневной ряд 17 тикеров ≤ 2021 (fastbt.data). Мутации № 21, 22 (окно с текущим днём) ловятся.
"""
import numpy as np
import pytest

from fastbt import fast_filtry_v5 as F
from fastbt import rangi_v6 as R
from fastbt.indicators import calc_adx, calc_atr
from fastbt.signals.v5_filtr import maska_atr_expand


def eq(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


def rnd_ohlc(seed, n=400):
    r = np.random.default_rng(seed)
    c = 100 + np.cumsum(r.normal(size=n))
    return c + np.abs(r.normal(size=n)), c - np.abs(r.normal(size=n)), c


@pytest.fixture(scope="module")
def real():
    from fastbt.data import TICKERS, daily_df, load_ticker
    assert daily_df()["date"].max().year <= 2021
    return [load_ticker(t, corp=True) for t in TICKERS]


# --- ADX
@pytest.mark.parametrize("seed", range(5))
def test_adx_sintetika(seed):
    h, l, c = rnd_ohlc(seed)
    assert eq(calc_adx(h, l, c), F.calc_adx_nb(h, l, c))


@pytest.mark.parametrize("n", [0, 1, 20, 28, 29, 30, 31])
def test_adx_korotkiy_ryad(n):
    h, l, c = rnd_ohlc(3, 40)
    assert eq(calc_adx(h[:n], l[:n], c[:n]), F.calc_adx_nb(h[:n], l[:n], c[:n]))


def test_adx_nulevoy_obyom_ploskiy_ryad():
    c = np.full(100, 50.0)
    assert eq(calc_adx(c, c, c), F.calc_adx_nb(c, c, c))


def test_adx_real(real):
    for t in real:
        assert eq(calc_adx(t.high, t.low, t.close), F.calc_adx_nb(t.high, t.low, t.close)), t.ticker


# --- маска ATR (расширяющееся окно до d-1)
@pytest.mark.parametrize("pct", [0, -5, 10, 30, 50, 80, 99.5, 100])
@pytest.mark.parametrize("seed", range(3))
def test_atr_sintetika(seed, pct):
    r = np.random.default_rng(seed)
    x = np.abs(r.normal(size=300)) + 1
    x[r.random(300) < 0.1] = np.nan
    x[x > 2.5] = 2.0                       # повторы значений: проверка bisect_right и равенств
    assert eq(maska_atr_expand(x, pct), F.maska_atr_expand_nb(x, pct))


def test_atr_pustaya_i_korotkaya_lenta():
    for n in (0, 1, 50, 51, 52):
        x = np.arange(1.0, n + 1)
        assert eq(maska_atr_expand(x, 30), F.maska_atr_expand_nb(x, 30))


def test_atr_real(real):
    for t in real:
        a = calc_atr(t.high, t.low, t.close, 14)
        for pct in (20, 50, 70):
            assert eq(maska_atr_expand(a, pct), F.maska_atr_expand_nb(a, pct)), (t.ticker, pct)


# --- ранги
def rnd_m(seed, nt=6, nd=400, p=0.5):
    r = np.random.default_rng(seed)
    m = r.normal(size=(nt, nd))
    m[r.random(m.shape) < p] = np.nan
    m[r.random(m.shape) < 0.1] = 0.0        # повторы: «строго меньше»
    return m


@pytest.mark.parametrize("tip", R.RANG_TIPY)
@pytest.mark.parametrize("seed", range(4))
def test_rang_sintetika(seed, tip):
    m = rnd_m(seed)
    assert eq(R.rang(m, tip), F.rang_nb(m, tip))


@pytest.mark.parametrize("tip", R.RANG_TIPY)
def test_rang_kraya(tip):
    for m in (np.full((3, 60), np.nan), np.zeros((3, 60)), rnd_m(1, 1, 30), rnd_m(2, 4, 0), rnd_m(3, p=0.95)):
        assert eq(R.rang(m, tip), F.rang_nb(m, tip))
    assert eq(R.rang(rnd_m(5), tip, okno=30, min_nabl=5), F.rang_nb(rnd_m(5), tip, okno=30, min_nabl=5))


def test_rang_oshibka_tipa():
    with pytest.raises(ValueError):
        F.rang_nb(rnd_m(0), "x")


def test_rang_real(real):
    m = np.full((len(real), 1500), np.nan)
    for k, t in enumerate(real):
        a = calc_atr(t.high, t.low, t.close, 14)[:1500] / t.close[:1500]
        m[k, :len(a)] = np.where(np.arange(len(a)) % 3 == 0, a, np.nan)
    for tip in R.RANG_TIPY:
        assert eq(R.rang(m, tip), F.rang_nb(m, tip)), tip


# --- причинность и мутации № 21, 22
def test_rang_bez_tekushchego_dnya():
    m = rnd_m(7)
    base = F.rang_nb(m, "srednee")
    d = 300
    m2 = m.copy()
    m2[:, d + 1:] = 123.0                          # будущее не влияет на день d
    assert eq(base[:, :d + 1], F.rang_nb(m2, "srednee")[:, :d + 1])
    m3 = m.copy()
    m3[:, d] = 1e6                                 # значение дня d не меняет окно дня d: ранг = 1 для кандидата
    r3 = F.rang_nb(m3, "obshchiy")[:, d]
    assert np.all(r3[np.isfinite(m3[:, d])] == 1.0)


def _rang_s_dnyom(m, okno=R.OKNO, min_nabl=R.MIN_NABL):
    """Мутант № 22: окно d-okno…d включает d (как uroven_v3.rang_pul, Ж-095)."""
    out = np.full(m.shape, np.nan)
    for d in range(m.shape[1]):
        w = m[:, max(0, d - okno):d + 1]
        v = np.sort(w[np.isfinite(w)])
        ok = np.isfinite(m[:, d])
        if len(v) >= min_nabl and ok.any():
            out[ok, d] = np.searchsorted(v, m[ok, d], side="left") / len(v)
    return out


def test_mutatsiya_22_rang_s_tekushchim_dnem_lovitsya():
    m = rnd_m(7)
    assert not eq(_rang_s_dnyom(m), F.rang_nb(m, "obshchiy"))
    assert eq(R.rang(m, "obshchiy"), F.rang_nb(m, "obshchiy"))


def _atr_s_dnyom(atr, pct):
    """Мутант № 21: порог по значениям ATR[0..d] (с текущим днём)."""
    out = np.zeros(len(atr), bool)
    for d in range(len(atr)):
        v = atr[:d + 1][~np.isnan(atr[:d + 1])]
        if len(v) > 51 and not np.isnan(atr[d]):
            out[d] = atr[d] > np.percentile(v, pct)
    return out


def test_mutatsiya_21_atr_s_tekushchim_dnem_lovitsya():
    r = np.random.default_rng(0)
    x = np.abs(r.normal(size=300)) + 1
    assert not eq(_atr_s_dnyom(x, 50), F.maska_atr_expand_nb(x, 50))
    assert eq(maska_atr_expand(x, 50), F.maska_atr_expand_nb(x, 50))


def test_atr_bez_tekushchego_dnya():
    r = np.random.default_rng(1)
    x = np.abs(r.normal(size=200)) + 1
    d = 150
    y = x.copy()
    y[d] = 1e9                                      # скачок в день d: порог дня d прежний, маска[d] = True
    assert F.maska_atr_expand_nb(y, 50)[d]
    y[d + 1:] = 1e9                                 # будущее не влияет на маску до d включительно
    assert eq(F.maska_atr_expand_nb(x, 50)[:d], F.maska_atr_expand_nb(y, 50)[:d])
