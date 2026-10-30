"""Комиссия книги МСП (#857, Ж-149, #861): акции дневные и оплачены в excess пары, не оплачен только хедж MIX. Синтетика."""
import numpy as np
import pytest

from fastbt import kniga_msp as km
from fastbt import tf_os_v5

C = 0.0004


def test_stavka_ta_zhe_chto_v_dvizhke():
    assert km.KOMISSIYA == tf_os_v5.KOMISSIYA == C


def test_dnevnye_stavki_excess_pary_uzhe_platit_polnyj_oborot():
    """Известный ответ: позиция 1 в день 1 и 2 в день 2 (множитель 1 -> 2), по два бара в день. Оборот бара:
    |Δpos| и закрытие на последнем баре -> [1, 1, 2, 2], издержки = C·6: каждый день вход и выход при СВОЁМ размере,
    ночью позиции нет (день 2 не зависит от дня 1). Значит, смена множителя отдельной сделки не порождает."""
    o = np.full(4, 100.0)
    den = np.array([0, 0, 1, 1])
    _, cost, turn = tf_os_v5.dohod(np.array([1, 1, 2, 2]), o, o, den)
    assert turn.tolist() == [1, 1, 2, 2] and cost.sum() == pytest.approx(C * 6)
    # линейность по размеру: excess·w — веса книги масштабируют и доходность, и издержки пары одинаково
    _, cost3, _ = tf_os_v5.dohod(np.array([3, 3, 3, 3]), o, o, den)
    _, cost1, _ = tf_os_v5.dohod(np.array([1, 1, 1, 1]), o, o, den)
    assert cost3.sum() == pytest.approx(3 * cost1.sum())


def test_hedzh_oblagaetsya():
    assert km.komissiya_mnozh(-0.4, -0.1) == pytest.approx(C * 0.3)
    assert km.komissiya_mnozh(0.0, 0.0) == 0.0


def _ist(T=40):
    return {"ex_sk": np.zeros((T, 2)), "ex_port": np.zeros(T), "ex_kniga": np.zeros(T), "imoex": np.zeros(T),
            "kvartal": np.zeros(T, int)}


ARGS = (np.array([5, 4, 3, -2, -1, 0.5]), np.full(6, 0.02), np.arange(6), np.array([0, 0, 0, 1, 1, 1]))


def test_kniga_dnya_mnozhitel_akcij_ne_platit():
    """Множитель 1 -> >1 меняет веса акций, комиссии книги нет (она уже в excess·w)."""
    ist = _ist()
    a = km.kniga_dnya(20, *ARGS, ist, k=6, rho=0.001)
    ist["ex_port"][:21] = np.where(np.arange(21) % 2, 1, -1) * 0.05 / np.sqrt(252)
    b = km.kniga_dnya(21, *ARGS, ist, k=6, rho=0.001)
    assert b.m != a.m and not np.allclose(a.u, b.u)
    assert b.komissiya == 0.0


def test_kniga_dnya_hedzh_platit_po_izmeneniyu_bety():
    """β книги меняется между днями (бета по 63 строкам до i-1) -> хедж меняется -> C·|Δ хеджа|."""
    rng = np.random.default_rng(5)
    T = 90
    ist = _ist(T)
    ist["imoex"] = rng.normal(0, 0.01, T)
    ist["ex_kniga"] = 1.5 * ist["imoex"] + rng.normal(0, 0.002, T)
    ist["ex_kniga"][:64] = 0.5 * ist["imoex"][:64]
    a = km.kniga_dnya(70, *ARGS, ist, k=6)
    b = km.kniga_dnya(71, *ARGS, ist, k=6)
    assert a.hedzh != b.hedzh
    assert b.komissiya == pytest.approx(C * abs(b.hedzh - a.hedzh)) and b.komissiya > 0


def test_mutaciya_komissiya_akcij_po_delta_ves_lovitsya(monkeypatch):
    """Мутант: комиссия по |Δ итогового веса акций| (двойной учёт) — тест выше обязан упасть (комиссия ≠ 0)."""
    ist = _ist()
    a = km.kniga_dnya(20, *ARGS, ist, k=6, rho=0.001)
    ist["ex_port"][:21] = np.where(np.arange(21) % 2, 1, -1) * 0.05 / np.sqrt(252)
    b = km.kniga_dnya(21, *ARGS, ist, k=6, rho=0.001)
    mutant = C * np.abs(b.u - a.u).sum()
    assert mutant > 0 and b.komissiya == 0.0
