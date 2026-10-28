"""fastbt.fdr: занятость ночи из сделок и позиций. Синтетика с известным ответом (контракт #376 §1.3 А, §1.5)."""
import numpy as np
import pytest

from fastbt import fdr as F
from fastbt.exec_e_types import Trade


def _tr(d, e, x, side=1):
    return Trade("A", d, e, 0, 100.0, exit_day=x, side=side)


def test_vnutridnevnaya_sdelka_noch_svobodna():
    tr = [_tr(2, 3, 3)]
    for r in F.REZHIMY[:1]:
        assert F.noch_iz_sdelok(tr, {2: 1.0}, 6, r).tolist() == [0] * 6


def test_pozitsiya_cherez_noch_zanyata_t0():
    out = F.noch_iz_sdelok([_tr(1, 2, 4)], {1: 0.5}, 6, "t0")
    assert out.tolist() == [0, 0, 0.5, 0.5, 0, 0]


def test_t0_stroka_vyhoda_svobodna():  # М-Ф2
    out = F.noch_iz_sdelok([_tr(1, 2, 4)], {1: 1.0}, 6, "t0")
    assert out[4] == 0.0


def test_t1_stroka_vyhoda_zanyata_sleduyushchaya_svobodna():  # М-Ф3
    out = F.noch_iz_sdelok([_tr(1, 2, 4)], {1: 1.0}, 6, "t1")
    assert out.tolist() == [0, 0, 1, 1, 1, 0]


def test_noch_vhoda_po_close_zanyata():  # М-Ф4
    for r in F.REZHIMY:
        assert F.noch_iz_sdelok([_tr(1, 2, 4)], {1: 1.0}, 6, r)[2] == 1.0


def test_otkrytaya_na_konce_okna_zanyata_do_konca():
    out = F.noch_iz_sdelok([_tr(1, 2, None)], {1: 1.0}, 5, "t1")
    assert out.tolist() == [0, 0, 1, 1, 1]


def test_t1_vnutridnevnaya_zanimaet_stroku_vyhoda():
    out = F.noch_iz_sdelok([_tr(1, 3, 3)], {1: 1.0}, 5, "t1")
    assert out.tolist() == [0, 0, 0, 1, 0]


def test_t1_min_edinitsa_pri_perevorote():
    tr = [_tr(0, 1, 3, 1), _tr(2, 3, 5, -1)]  # лонг закрыт и шорт открыт по close 3
    out = F.noch_iz_sdelok(tr, {0: 1.0, 2: 1.0}, 6, "t1")
    assert out[3] == 1.0  # не 2
    assert F.noch_iz_sdelok(tr, {0: 1.0, 2: 1.0}, 6, "t0")[3] == 1.0


def test_net_vesa_oshibka():
    with pytest.raises(KeyError):
        F.noch_iz_sdelok([_tr(1, 2, 3)], {}, 5, "t0")


def test_nepravilnyj_rezhim_i_pustaya_lenta():
    with pytest.raises(ValueError):
        F.noch_iz_sdelok([], {}, 3, "t2")
    assert F.noch_iz_sdelok([], {}, 3, "t1").tolist() == [0, 0, 0]
    assert F.noch_iz_pos([], [], "t1").size == 0


def test_pos_t0_ravno_abs():
    pos = np.array([0.0, 0.5, -0.5, 0.0])
    assert F.noch_iz_pos(pos, np.array([1, 0, 0, 0], bool), "t0").tolist() == [0, 0.5, 0.5, 0]


def test_pos_t1_zakryto_perevorot_i_sokrashchenie():
    pos = np.array([1.0, 0.4, -1.0, 0.0])
    f = np.array([1, 0, 0, 0], bool)
    # t1: 1; 0.4+0.6; -1: знак сменился → 1+0.4 → min 1; 0: закрыто 1 → 1
    assert F.noch_iz_pos(pos, f, "t1").tolist() == [1.0, 1.0, 1.0, 1.0]
    pos2 = np.array([0.0, 0.5, 0.2, 0.2])
    out = F.noch_iz_pos(pos2, np.array([1, 0, 0, 0], bool), "t1")
    assert out.tolist() == pytest.approx([0.0, 0.5, 0.5, 0.2])


def test_pos_pervaya_stroka_pary_bez_zakryto():
    pos = np.array([0.7, 0.0, 0.3])  # строка 2 — начало новой пары: prev=0 не учитывается
    out = F.noch_iz_pos(pos, np.array([1, 0, 1], bool), "t1")
    assert out.tolist() == pytest.approx([0.7, 0.7, 0.3])
