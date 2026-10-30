"""Проверка тестировщика #770 (сверх тестов автора): граница «≥» и «>» на точном пределе, правило не в canon_v5, устойчивый отбор правил-отказов."""
import numpy as np
import pytest
from test_canon_bb_predel import KUB, PARY, T, signaly
from test_canon_nesovm import baza
from test_canon_nesovm_orakul import otkazy_pravila

from fastbt import canon_v5 as c5, canon_v6_nesovm as nsv


def test_v_setke_net_tochnogo_ravenstva_k_predelu():
    """Следствие для мутации «> вместо ≥»: на сетке S2 она неотличима (ни одна пара не лежит ровно на пределе) — граница проверяется отдельно ниже."""
    assert all(k != nsv.predel_z(int(w)) for w, k in PARY)


@pytest.mark.xfail(strict=True, reason="находка #770: при k == (w−1)/√w сигнал достижим (w−1 равных цен и скачок), а правило ≥ точку выбрасывает; "
                                      "нужно «>». На сетке равенства нет — числа не меняются.")
@pytest.mark.parametrize("w", [5, 25, 100])
def test_na_tochnom_predele_signal_est_i_pravilo_ne_vybrasyvaet(w):
    k = nsv.predel_z(w)
    c = np.full(T, 100.0)
    c[300::300] += 40.0
    assert signaly(c, w, k) > 0
    assert not nsv.bb_predel("S2", dict(ma_tip="sma", bb_window=w, bb_std=k))


def test_pravilo_ne_v_canon_v5():
    """canon_v5 заморожен: точка за пределом остаётся каноном v5 (правило 9 живёт только в canon_v6)."""
    assert c5.canon_v5("S2", baza("S2", ma_tip="sma", bb_window=7, bb_std=4.0)) is not None


def test_otbor_pravil_otkazov_po_tozhdestvu_a_ne_po_indeksu():
    assert nsv.bb_predel in nsv.PRAVILA and nsv.bb_predel not in otkazy_pravila()
    assert len(otkazy_pravila()) == len(nsv.PRAVILA) - 1 and KUB["sid"] == "S2"
