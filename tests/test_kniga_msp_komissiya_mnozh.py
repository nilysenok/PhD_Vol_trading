"""Комиссия за оборот, вызванный весами Sk, нетто, сектором и множителями книги МСП (#857, Ж-149). Синтетика; без сети."""
import numpy as np
import pytest

from fastbt import kniga_msp as km

C = 0.0004
T0 = np.array([0, 1])           # два столбца -> тикеры 0 и 1


def _kom(w, wp, raw, rawp, tik=T0, h=0.0, hp=0.0):
    a = (np.array(x, float) for x in (w, wp, raw, rawp))
    return km.komissiya_mnozh(*a, tik, h, hp)


def test_stavka_ta_zhe_chto_v_dvizhke():
    from fastbt.tf_os_v5 import KOMISSIYA
    assert km.KOMISSIYA == KOMISSIYA == C


def test_mnozhitel_1_2_1_platit_dva_raza():
    """Сырая позиция 0,5 постоянна, коэффициент цепочки 1 -> 2 -> 1: оборот 0,5 и 0,5."""
    assert _kom([1.0, 0], [0.5, 0], [0.5, 0], [0.5, 0]) == pytest.approx(C * 0.5)
    assert _kom([0.5, 0], [1.0, 0], [0.5, 0], [0.5, 0]) == pytest.approx(C * 0.5)


def test_a_den_bez_signalov_menyaetsya_ves_sk():
    assert _kom([0.7, 0], [0.5, 0], [0.5, 0], [0.5, 0]) == pytest.approx(C * 0.2)


def test_b_den_bez_signalov_srabotal_netto_ili_sektor():
    """Сырые позиции те же, итоговый вес первой ноги срезан лимитом 1,0 -> 0,6."""
    assert _kom([0.6, -0.5], [1.0, -0.5], [1.0, -0.5], [1.0, -0.5]) == pytest.approx(C * 0.4)


def test_vhod_sdelki_ne_oblagaetsya_dvazhdy():
    assert _kom([1.0, 0], [0, 0], [0.5, 0], [0, 0]) == pytest.approx(0.0)


def test_vyhod_sdelki_ne_oblagaetsya_dvazhdy():
    assert _kom([0, 0], [1.0, 0], [0, 0], [0.5, 0]) == pytest.approx(0.0)


def test_plecho_imeet_znachenie():
    assert _kom([1.5, 0], [0.5, 0], [0.5, 0], [0.5, 0]) > _kom([1.0, 0], [0.5, 0], [0.5, 0], [0.5, 0])


def test_stolbcy_odnogo_tikera_skladyvayutsya():
    assert _kom([1.0, -1.0], [0.5, -0.5], [0.5, -0.5], [0.5, -0.5], tik=np.array([0, 0])) == 0.0


def test_hedzh_oblagaetsya():
    assert _kom([0, 0], [0, 0], [0, 0], [0, 0], h=-0.4, hp=-0.1) == pytest.approx(C * 0.3)


def _ist(T=40):
    return {"ex_sk": np.zeros((T, 2)), "ex_port": np.zeros(T), "ex_kniga": np.zeros(T), "imoex": np.zeros(T),
            "kvartal": np.zeros(T, int)}


ARGS = (np.array([5, 4, 3, -2, -1, 0.5]), np.full(6, 0.02), np.arange(6), np.array([0, 0, 0, 1, 1, 1]))


def test_kniga_dnya_mnozhitel_spisyvaet():
    ist = _ist()
    a = km.kniga_dnya(20, *ARGS, ist, k=6, rho=0.001)
    ist["ex_port"][:21] = np.where(np.arange(21) % 2, 1, -1) * 0.05 / np.sqrt(252)
    b = km.kniga_dnya(21, *ARGS, ist, k=6, rho=0.001)
    assert b.m != a.m
    assert b.komissiya == pytest.approx(C * np.abs(b.u - a.u).sum()) and b.komissiya > 0


def _dva_dnya_smena_sk():
    ist = _ist()
    ist["ex_sk"][:20, 0] = np.where(np.arange(20) % 2, 1, -1) * 0.001
    ist["ex_sk"][:20, 1] = np.where(np.arange(20) % 2, 1, -1) * 0.001
    a = km.kniga_dnya(20, *ARGS, ist, k=6, rho=0.001)
    ist["ex_sk"][20, 0] = 0.05                          # σ̂ первого Sk выросла -> веса Sk другие, сигналов нет
    return a, km.kniga_dnya(21, *ARGS, ist, k=6, rho=0.001)


def test_kniga_dnya_ves_sk_spisyvaet():
    a, b = _dva_dnya_smena_sk()
    assert b.m == a.m == 1.0 and not np.allclose(a.w_sk, b.w_sk)
    assert b.komissiya == pytest.approx(C * np.abs(b.u - a.u).sum()) and b.komissiya > 0


def test_mutaciya_netto_ves_sk_v_syroj_pozicii_lovitsya(monkeypatch):
    """Мутант: «сырой» позицией считается итоговая (веса Sk/нетто/лимиты не облагаются) — тест (а) обязан упасть."""
    orig = km.komissiya_mnozh
    monkeypatch.setattr(km, "komissiya_mnozh", lambda w, wp, raw, rawp, *a: orig(w, wp, w, wp, *a))
    _, b = _dva_dnya_smena_sk()
    assert b.komissiya == 0.0                           # мутант не списывает ничего
    monkeypatch.undo()
    _, b2 = _dva_dnya_smena_sk()
    assert b2.komissiya > 0                             # правка списывает
