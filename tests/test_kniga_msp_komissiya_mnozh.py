"""Комиссия за оборот, вызванный множителями книги МСП (находка #857, Ж-149). Синтетика, руками; без сети."""
import numpy as np
import pytest

from fastbt import kniga_msp as km

C = 0.0004
T0 = np.array([0, 1])           # два столбца (Sk) -> тикеры 0 и 1


def _kom(w, wp, u0, u0p, mf, tik=T0, h=0.0, hp=0.0):
    return km.komissiya_mnozh(np.array(w, float), np.array(wp, float), np.array(u0, float), np.array(u0p, float),
                              mf, tik, h, hp)


def test_stavka_ta_zhe_chto_v_dvizhke():
    from fastbt.tf_os_v5 import KOMISSIYA
    assert km.KOMISSIYA == KOMISSIYA == C


def test_mnozhitel_1_2_1_platit_dva_raza():
    """Позиция пары постоянна (u0=0,5), множитель 1 -> 2 -> 1: оборот |0,5| и |−0,5|."""
    assert _kom([1.0, 0], [0.5, 0], [0.5, 0], [0.5, 0], 2.0) == pytest.approx(C * 0.5)
    assert _kom([0.5, 0], [1.0, 0], [0.5, 0], [0.5, 0], 1.0) == pytest.approx(C * 0.5)


def test_den_bez_signalov_tolko_mnozhitel_spisyvaet():
    assert _kom([0.3, -0.2], [0.15, -0.1], [0.15, -0.1], [0.15, -0.1], 2.0) > 0


def test_vhod_sdelki_ne_oblagaetsya_dvazhdy():
    """Новая ставка 0 -> 0,5 при m=2: вход уже в excess пары (масштабирован) — сверх него ничего."""
    assert _kom([1.0, 0], [0, 0], [0.5, 0], [0, 0], 2.0) == pytest.approx(0.0)


def test_vyhod_sdelki_ne_oblagaetsya_dvazhdy():
    assert _kom([0, 0], [1.0, 0], [0, 0], [0.5, 0], 2.0) == pytest.approx(0.0)


def test_plecho_imeet_znachenie():
    """Комиссия растёт с множителем (3 против 2 при той же базе)."""
    assert _kom([1.5, 0], [0.5, 0], [0.5, 0], [0.5, 0], 3.0) > _kom([1.0, 0], [0.5, 0], [0.5, 0], [0.5, 0], 2.0)


def test_stolbcy_odnogo_tikera_skladyvayutsya():
    """Два Sk с противоположными ставками на один тикер: нетто-оборот, а не сумма модулей."""
    assert _kom([1.0, -1.0], [0.5, -0.5], [0.5, -0.5], [0.5, -0.5], 2.0, tik=np.array([0, 0])) == 0.0


def test_hedzh_oblagaetsya():
    assert _kom([0, 0], [0, 0], [0, 0], [0, 0], 1.0, h=-0.4, hp=-0.1) == pytest.approx(C * 0.3)


def test_kniga_dnya_spisyvaet_na_dne_bez_signalov_izmeneniya():
    """Те же скоры два дня подряд: m 1 -> >1 меняет только множитель; Itog.komissiya = C·Σ|Δ итоговых весов|."""
    n, T = 6, 40
    score, sigma = np.array([5, 4, 3, -2, -1, 0.5]), np.full(n, 0.02)
    sek, sk = np.arange(n), np.zeros(n, int)
    ist = {"ex_sk": np.zeros((T, 1)), "ex_port": np.zeros(T),
           "ex_kniga": np.zeros(T), "imoex": np.zeros(T), "kvartal": np.zeros(T, int)}
    a = km.kniga_dnya(20, score, sigma, sek, sk, ist, k=6, rho=0.001)
    ist["ex_port"][:21] = np.where(np.arange(21) % 2, 1, -1) * 0.05 / np.sqrt(252)
    b = km.kniga_dnya(21, score, sigma, sek, sk, ist, k=6, rho=0.001)
    assert b.m != a.m
    assert b.komissiya == pytest.approx(C * np.abs(b.u - a.u).sum()) and b.komissiya > 0
