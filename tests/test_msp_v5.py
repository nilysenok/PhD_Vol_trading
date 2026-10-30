"""Ф-1: драйвер МСП `fastbt.msp_v5` на синтетике (без сети, данных нет, период 2016–2017).

Гейты: λ = 0 ≡ запуск без фаз побайтно; фаза на день d — по данным ≤ d−1 (мутация «фаза с днём d», `fazy._LAG = 0`,
обязана ронять тест); история книги не читает доходность дня решения и позже; ≥ 2020 — отказ.
"""
import numpy as np
import pandas as pd
import pytest

from fastbt import fazy, msp_v5, reyting

T, D0 = 330, 300
IDX = pd.bdate_range("2016-01-04", periods=T)
TIK = ["A", "B", "C"]
SEKTOR = {"A": "X", "B": "X", "C": "Y"}


def _dannye(krah: bool = False) -> dict:
    rng = np.random.default_rng(1)
    r = 0.002 + 0.004 * rng.standard_normal((T, 3))
    akc = pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=IDX, columns=TIK)
    im = pd.Series(100 * np.cumprod(1 + 0.002 + 0.004 * rng.standard_normal(T)), index=IDX)
    if krah:                                     # день D0: обвал; его не должно быть видно в фазе дня D0
        akc.iloc[D0] *= 0.5
        im.iloc[D0] *= 0.5
    return {"imoex": im, "akcii": akc, "karta": SEKTOR}


def _sig():
    rng = np.random.default_rng(2)
    f = lambda v: pd.DataFrame(v, index=IDX, columns=TIK)           # noqa: E731
    return {"S1": {"storona": f(np.ones((T, 3))), "sila": f(rng.uniform(0, 1, (T, 3))), "g": f(np.ones((T, 3)))},
            "S7": {"storona": f(-np.ones((T, 3))), "sila": f(rng.uniform(0, 1, (T, 3))), "g": f(np.ones((T, 3)))}}


def _vhod():
    rng = np.random.default_rng(3)
    f = lambda v: pd.DataFrame(v, index=IDX, columns=TIK)           # noqa: E731
    return (f(rng.uniform(0.01, 0.03, (T, 3))), f(0.01 * rng.standard_normal((T, 3))),
            pd.Series(0.005 * rng.standard_normal(T), index=IDX))


def _run(lam, dannye=None, dohod=None, k=6):
    sigma, d, im = _vhod()
    rb = d if dohod is None else dohod                           # контракт Ж-149: R_brutto + T (T=0 — как прежде)
    return msp_v5.progon(_sig(), sigma, SEKTOR, {"R_brutto": rb, "T": rb * 0.0}, im, lam, dannye, k=k)


def _b(r):
    return r.u.to_numpy().tobytes() + r.hedzh.to_numpy().tobytes() + r.ex_kniga.to_numpy().tobytes()


def test_lam0_pobaytovo_ravno_bez_faz():
    s = _sig()
    r0, bez = _run(0.0, _dannye()), _run(0.0, None)
    assert _b(r0) == _b(bez)
    for sk, v in s.items():                      # «без фаз» = сторона · ранг · g, руками
        ruki = v["storona"] * reyting.rang_sily(v["sila"]) * v["g"]
        pd.testing.assert_frame_equal(r0.skory[sk], ruki, check_exact=True)


def test_lam1_otlichaetsya_ot_lam0():              # тест не пустой: фазы реально меняют скоры и книгу
    r0, r1 = _run(0.0, _dannye()), _run(1.0, _dannye())
    assert _b(r0) != _b(r1)
    assert not r0.skory["S1"].equals(r1.skory["S1"])


def _skor_d0(dannye):
    fr, fs = reyting.fazy_dnya(TIK, IDX, dannye)
    return msp_v5.skory(_sig(), fr, fs, 1.0)["S1"].iloc[D0]


def test_faza_bez_dnya_d():
    a, b = _skor_d0(_dannye()), _skor_d0(_dannye(krah=True))
    pd.testing.assert_series_equal(a, b)         # обвал в день D0 на скор дня D0 не влияет


def test_mutatsiya_faza_s_dnyom_d_lovitsya(monkeypatch):
    monkeypatch.setattr(fazy, "_LAG", 0)
    a, b = _skor_d0(_dannye()), _skor_d0(_dannye(krah=True))
    assert not a.equals(b)                       # на ошибке (фаза с днём d) предыдущий тест упал бы


def test_istoriya_ne_chitaet_dohod_dnya_resheniya():
    sigma, d, im = _vhod()
    d2 = d.copy()
    d2.iloc[290:] = 0.5                          # доходность с дня 290 и позже подменена
    a, b = _run(1.0, _dannye(), d), _run(1.0, _dannye(), d2)
    np.testing.assert_array_equal(a.u.iloc[:291].to_numpy(), b.u.iloc[:291].to_numpy())
    assert not np.array_equal(a.u.iloc[291:].to_numpy(), b.u.iloc[291:].to_numpy())


def test_dannye_2020_otkaz():
    idx = pd.bdate_range("2019-12-20", periods=10)
    f = pd.DataFrame(1.0, index=idx, columns=["A"])
    with pytest.raises(ValueError):
        msp_v5.progon({"S1": {"storona": f, "sila": f, "g": f}}, f, {"A": "X"}, f, pd.Series(0.0, index=idx), 0.0)
