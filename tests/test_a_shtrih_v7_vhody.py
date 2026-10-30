"""Л3.3-прогон: сборка входов A′-расширенной из исполнителя голой идеи (fastbt/a_shtrih_v7_vhody). Синтетика, без данных.
check_* — на правильном коде; test_mut_* — подложенная ошибка (check обязан упасть). Мутации: M-d1 (сигнал сдвинут на день назад —
вход в день сигнала), позиция без знака (|poz|), дыра цены заполнена нулём вместо NaN."""
import numpy as np
import pandas as pd
import pytest

from fastbt import a_shtrih_v7_vhody as V

T, N = 12, 3
DATY = pd.bdate_range("2016-03-01", periods=T)


def lenta(seed=1):
    rng = np.random.default_rng(seed)
    c = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (T, N)), axis=0), index=DATY, columns=list("ABC"))
    o = c.shift(1).bfill() * (1 + rng.normal(0, 0.002, (T, N)))
    return dict(daty=DATY, tk=list("ABC"), o=o, cr=c, div=pd.DataFrame(0.0, index=DATY, columns=list("ABC")), stavka=np.full(T, 0.07))


def signaly():
    s = np.zeros((T, N), np.int8)
    s[1, 0], s[2, 0] = 1, 1                  # A: лонг с сигналом в дни 1, 2
    s[3, 1] = -1                             # B: шорт
    s[4, 0], s[5, 0] = -1, 0                 # A: смена знака в день 4 (разворот по открытию)
    return s


def vypolnit(D, s, **kw):
    return V.pary(D, s, vhod="d1", tau=2, sost=False, **kw)


def check_znak_i_dzn():
    D, s = lenta(), signaly()
    X, P = vypolnit(D, s)
    assert X.shape == P.shape == (N, T)
    assert P[1].min() == -1 and P[1].max() == 0       # B: шорт — знак минус, не |позиция|
    assert (P[0] >= 0).sum() > 0 and P[0].max() == 1
    assert P[0, 1] == 0 and P[1, 3] == 0              # M-d1: в день сигнала позиции нет (вход d+1)
    assert X[1, 3] == 0                               # и результата нет
    assert np.isfinite(X).all()


def check_dyra_cenny_nan():
    D, s = lenta(), signaly()
    D["o"].iloc[7, 2] = np.nan                        # у C нет цены в день 7
    X, P = vypolnit(D, s)
    assert np.isnan(X[2, 7]) and P[2, 7] == 0 and np.isfinite(X[2, 6])


def check_summa_ravna_gv_schet():
    """Среднее по тикерам X == excess портфеля golaya_v7.schet (вес 1/N); состояние — разворот в один open не теряет и не удваивает."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
    import golaya_v7 as gv
    D, s = lenta(), signaly()
    for kw in (dict(tau=2, sost=False), dict(tau=None, sost=True)):
        X, _ = V.pary(D, s, vhod="d1", **kw)
        ref = gv.schet(D, s, "d1", **kw)["excess"]
        assert np.allclose(X.sum(0) / N, ref, atol=1e-12), kw


def test_znak_i_den_signala():
    check_znak_i_dzn()


def test_dyra_ceny():
    check_dyra_cenny_nan()


def test_summa_ravna_gv_schet():
    check_summa_ravna_gv_schet()


def test_razvorot_poslednyaya_poziciya_pobezhdaet():
    D = lenta()
    s = np.zeros((T, N), np.int8)
    s[1, 0], s[2, 0], s[3, 0] = 1, 1, 0              # лонг: сигнал дней 1–2 (состояние), закрытие по смене знака
    s[3, 0], s[4, 0] = -1, -1
    _, P = V.pary(D, s, vhod="d1", tau=None, sost=True)
    assert P[0, 4] == -1 and P[0, 2] == 1             # день разворота — новая сторона (не сумма 0)


def test_mut_m_d1_lovitsya():
    """Мутация: сигнал сдвинут на день назад (вход в день сигнала) — check обязан упасть."""
    D, s = lenta(), signaly()
    X, P = vypolnit(D, np.roll(s, -1, axis=0))
    with pytest.raises(AssertionError):
        assert P[0, 1] == 0 and X[0, 1] == 0


def test_mut_poziciya_bez_znaka_lovitsya():
    D, s = lenta(), signaly()
    _, P = vypolnit(D, s)
    with pytest.raises(AssertionError):
        assert np.abs(P)[1].min() == -1


def test_mut_dyra_nulyom_lovitsya():
    D, s = lenta(), signaly()
    D["o"].iloc[7, 2] = np.nan
    X, _ = vypolnit(D, s)
    with pytest.raises(AssertionError):
        assert np.isnan(np.nan_to_num(X)[2, 7])                   # дыра, заполненная нулём, check не проходит


def test_d1_maksimum_na_lage_0_i_sdvig_lovitsya():
    rng = np.random.default_rng(5)
    n = 120
    D = dict(lenta(3), daty=pd.bdate_range("2016-03-01", periods=n))
    c = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (n, N)), axis=0), index=D["daty"], columns=list("ABC"))
    D.update(cr=c, o=c.shift(1).bfill(), div=pd.DataFrame(0.0, index=D["daty"], columns=list("ABC")), stavka=np.full(n, 0.07))
    s = (rng.random((n, N)) < 0.15).astype(np.int8) * rng.choice([-1, 1], (n, N)).astype(np.int8)
    X, P = V.pary(D, s, vhod="d1", tau=2, sost=False)
    r = V.proverka_sdviga(X, P, c.to_numpy())
    assert max(r, key=lambda k: abs(r[k])) == 0
    with pytest.raises(ValueError):                                   # мутация: ряд результата сдвинут на день
        V.proverka_sdviga(np.roll(X, 1, axis=1), P, c.to_numpy())


def test_otkaz_pozicii_net():
    D, s = lenta(), signaly()
    with pytest.raises(ValueError):
        V.pary(D, np.full((T, N + 1), 2, np.int8), vhod="d1", tau=2, sost=False)
