"""Правки по вердикту проверяющего PR #506: потолок ставки, дыра/ноль OPEN в MIX, просадка от пика, порядок §11.2."""
import numpy as np
import pandas as pd
import pytest

from fastbt import kniga_msp as km
from fastbt import mix_skleyka as mx


def test_stavka_ne_vyshe_100_procentov_nav():
    # σ_d = 0,001 < ρ = 0,01: без потолка u = 10 (1000 % NAV)
    u = km.vybor_stavok([2.0, -1.0], [0.001, 0.001], [0, 1], 2, 0.01)
    assert u == pytest.approx([1.0, -1.0])
    u2 = km.vybor_stavok([2.0], [0.02], [0], 1, 0.01)           # ниже потолка — без изменений
    assert u2 == pytest.approx([0.5])


def _kontrakt(opn):
    daty = pd.bdate_range("2016-02-01", "2016-02-12")
    df = pd.DataFrame({"OPEN": opn}, index=daty)
    return {"MXH6_2016": df}, daty


def test_nol_v_open_dayot_oshibku_a_ne_minus_100():
    opn = [100.0 + i for i in range(10)]
    opn[4] = 0.0
    k, daty = _kontrakt(opn)
    with pytest.raises(ValueError):                              # ноль на t+1 раньше давал ret = −100 %
        mx.skleit(k, daty)
    opn[4], opn[3] = 104.0, 0.0                                  # ноль на t
    with pytest.raises(ValueError):
        mx.skleit(*_kontrakt(opn))


def test_dyra_v_open_dayot_nan_a_ne_minus_100():
    opn = [100.0 + i for i in range(10)]
    k, daty = _kontrakt(opn)
    k["MXH6_2016"] = k["MXH6_2016"].drop(daty[4])                # дыры нет в данных вовсе
    r = mx.skleit(k, daty)
    assert np.isnan(r["ret_oo"].iloc[3]) and np.isnan(r["ret_oo"].iloc[4])
    assert r["ret_oo"].iloc[2] == pytest.approx(103 / 102 - 1)
    assert (r["ret_oo"].dropna() > -0.5).all()


def test_prosadka_ot_pika_a_ne_ot_nulya():
    ex = np.zeros(40)
    ex[2], ex[5] = 0.10, -0.06                                   # от пика 0,10 -> 0,04: 6 % > 5 %; от нуля было бы 4 %
    assert km.vyklyuchatel(ex, 20) == 0.5
    ex[2] = 0.0                                                  # без пика, −4 % от нуля
    ex[5] = -0.04
    assert km.vyklyuchatel(ex, 20) == 1.0                        # 4 % < 5 %


def test_porjadok_11_2_vesa_sk_do_netto():
    T, S = 330, 2
    rng = np.random.default_rng(3)
    ist = {"ex_sk": np.stack([rng.normal(0.0005, 0.005, T), rng.normal(0.0005, 0.03, T)], axis=1),
           "ex_port": rng.normal(0, 0.005, T), "ex_kniga": rng.normal(0, 0.005, T),
           "imoex": rng.normal(0, 0.01, T), "kvartal": np.arange(T) // 63}
    i = 300
    score, sigma, sektor, sk = [3.0, 2.0, -1.0], np.full(3, 0.02), [0, 1, 2], [0, 0, 1]
    it = km.kniga_dnya(i, score, sigma, sektor, sk, ist, k=3)
    u0 = km.vybor_stavok(score, sigma, sektor, 3, 0.01)
    mult = (S * it.w_sk)[np.asarray(sk)]
    pravilno = km.netto(u0 * mult)
    ne_tak = km.netto(u0) * mult                                 # веса после нетто
    assert not np.allclose(pravilno, ne_tak)                     # тест не пустой: порядок различим
    assert it.u == pytest.approx(pravilno * it.m * it.vykl)
    assert it.u != pytest.approx(ne_tak * it.m * it.vykl)
