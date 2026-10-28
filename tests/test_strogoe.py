"""Свои тесты CE-1 / CE-4 (fastbt/strogoe.py, fastbt/ledger_ce.py): две пары, o10_ryad, nogi с дивидендом, писатель ce-ledger.
Синтетика, данные не читаются."""
import numpy as np
import pandas as pd
import pytest

from fastbt import ledger_ce, strogoe
from fastbt.exec_e import Day
from fastbt.s9 import Engine

C = 0.0004


def eng2():
    """Две пары по 4 даты (ср…пн через выходные не нужны): строки пары 0 — 0..3, пары 1 — 4..7."""
    dates = pd.bdate_range("2019-03-04", periods=4)
    e = Engine.__new__(Engine)
    e.r = {"date": np.r_[np.arange(4), np.arange(4)], "tick": np.r_[np.zeros(4, int), np.ones(4, int)], "bar": np.r_[np.arange(4), np.arange(4)]}
    e.first = np.array([1, 0, 0, 0, 1, 0, 0, 0], bool)
    e.dates, e.cnt = dates, np.full(4, 2.0)
    e.kr, e.rf, e.cal = np.zeros(4), np.zeros(4), np.ones(4)
    e.lr_row = np.zeros(8)
    return e


def test_dve_pary_ne_tekut_cherez_granicu():
    e = eng2()
    nr, dr = np.full(8, 0.01), np.full(8, 0.02)
    pos = np.array([1, 1, 1, 1, 0, 0, 0, 0.0])                   # у пары 0 pos[T]=1 не исполняется; пара 1 не наследует prev
    p = strogoe.port_strogiy(e, pos, C, zaem=0.0, nogi=(nr, dr))
    g = np.array([0.02, 0.03, 0.03, 0.01])                       # строка 0: pos[0]·dn; 1,2: n+dn; 3 — только ночь
    assert np.allclose(p["gross"].values, g / 2)
    assert np.allclose(p["turnover"].values, np.array([1, 0, 0, 1]) / 2), "вход в t0, выход на границе окна"


def test_zaem_krome_nochi_ne_beretsya_i_delitsya_na_cnt():
    e = eng2()
    pos = np.array([-1, -1, 0, 0, 0, 0, 0, 0.0])
    p = strogoe.port_strogiy(e, pos, 0.0, zaem=0.10, nogi=(np.zeros(8), np.zeros(8)))
    assert list(p["zaem"].values) == [0.0, 0.10 / 365 / 2, 0.10 / 365 / 2, 0.0]
    assert np.allclose(p["r"].values, -p["zaem"].values), "плата входит в r"
    assert p["noch"].values[0] == 0.0


def test_mutatsiya_zaem_po_pos_t_krasnaya():
    """М-CE7: плата по pos[t] (а не pos[t−1]) дала бы z[0] > 0 для шорта, открытого на close 0."""
    e = eng2()
    pos = np.array([-1, 0, 0, 0, 0, 0, 0, 0.0])
    z = strogoe.port_strogiy(e, pos, C, zaem=0.05, nogi=(np.zeros(8), np.zeros(8)))["zaem"].values
    assert z[0] == 0.0 and z[1] > 0.0


def test_bez_nog_otkaz():
    with pytest.raises(ValueError):
        strogoe.port_strogiy(eng2(), np.zeros(8), C)


def test_patched_snimaet_podmenu_dazhe_pri_oshibke():
    e = eng2()
    with pytest.raises(RuntimeError):
        with strogoe.patched(e, (np.zeros(8), np.zeros(8))):
            raise RuntimeError
    assert "port" not in vars(e)


def mk(o10, c):
    t = np.arange(590, 590 + 5 * 12, 5)
    o = np.full(12, c)
    o[2] = o10
    return Day(o, o, o, np.full(12, c), False, t=t)


def test_o10_ryad_i_nogi_s_dividendom():
    e = eng2()
    days = [[mk(100, 100), mk(110, 105), mk(106, 107), mk(107, 108)]] * 2
    o10, inf = strogoe.o10_ryad(e, days)
    assert np.allclose(o10[0, :3], [110, 106, 107]) and np.isnan(o10[0, 3]) and inf["podmen"] == 0
    close = np.array([[100, 105, 107, 108.0]] * 2)
    div = np.zeros((2, 4))
    div[:, 1] = 5.0                                              # гэп в день 1: дивиденд попадает в ночную ногу строки 0
    n, dn, inf = strogoe.nogi(e, o10, div, close)
    assert inf["n_dyr"] == 0
    assert n[0] == pytest.approx(np.log((110 + 5 * 0.87) / 100)) and dn[0] == pytest.approx(np.log(105 / 110))
    assert n[1] == pytest.approx(np.log(106 / 105)) and n[3] == 0.0 and dn[3] == 0.0, "последняя строка пары: цен 'после' нет"


def test_dyra_v_cene_schitaetsya_i_porog_otkazyvaet():
    """Дыра не молчит (Д4): n_dyr > 0, нога дыры 0 с отметкой; много дыр — ValueError. Мутация nan_to_num: n_dyr бы остался 0."""
    e = eng2()
    o10 = np.array([[110, 106, 107, np.nan]] * 2)
    close = np.array([[100, 105, 107, 108.0]] * 2)
    o10[1, 1] = np.nan                                           # дыра в тикере 1, строка 5
    div = np.zeros((2, 4))
    n, dn, inf = strogoe.nogi(e, o10, div, close, max_dyr=0.5)
    assert inf["n_dyr"] == 1 and n[5] == 0.0 and dn[5] == 0.0
    assert n[1] != 0.0 and not np.isnan(n).any()
    with pytest.raises(ValueError):
        strogoe.nogi(e, o10, div, close)                         # 1 из 4 строк тикера > 1 %


def test_o10_zapasnoe_pravilo_schitaet_podmeny():
    t = np.arange(605, 605 + 5 * 12, 5)                          # бара ровно в 10:00 нет — первый позже
    d = Day(np.arange(12.0) + 1, np.zeros(12), np.zeros(12), np.zeros(12), False, t=t)
    o10, inf = strogoe.o10_ryad(eng2(), [[mk(1, 1), d, d]])
    assert o10[0, 0] == 1.0 and inf["podmen"] >= 1


def test_pisatel_ce_osnovnaya_iz_parametrov_i_chitatel(tmp_path):
    kl = {"family": "C1", "komissiya": "0.0004", "zaem": "0.05", "ispolnenie": dict(strogoe.ISPOLNENIE)}
    w = ledger_ce.Pisatel_ce(tmp_path, "kod1", "vh1")
    r = w.zapisat({"semya": "C", "klyuch": kl, "period": "train_ce", "by_year": {"2018": 1.0, "2019": 2.0}, "osnovnaya": False})
    assert r["osnovnaya"] is True, "основная — функция параметров, чужое значение перезаписано"
    kl2 = dict(kl, ispolnenie=strogoe.ispolnenie_tochki("E", dict(family="E1", f="zakrytie_d1")))
    r2 = w.zapisat({"semya": "E", "klyuch": kl2, "period": "train_ce"})
    assert r2["osnovnaya"] is False and r2["klyuch_sha"] != r["klyuch_sha"]
    assert len(ledger_ce.Chitatel_ce().chitat(tmp_path)) == 2


def test_klyuch_sha_chuvstvuet_k_ispolneniyu_komissii_i_zaemu():
    """Д5 (mutations № 7, 38): ispolnenie, комиссия, ставка шорта — в ключе."""
    kl = {"komissiya": "0.0004", "zaem": "0.05", "ispolnenie": dict(strogoe.ISPOLNENIE)}
    base = ledger_ce.klyuch_sha(kl, "k", "v")
    for ch in (dict(kl, komissiya="0.00035"), dict(kl, zaem="0.10"),
               dict(kl, ispolnenie=dict(kl["ispolnenie"], vyhod="close_x"))):
        assert ledger_ce.klyuch_sha(ch, "k", "v") != base
    assert ledger_ce.klyuch_sha(kl, "k2", "v") != base


def test_chitatel_otkazyvaet_god_vne_okna_i_raznye_kod(tmp_path):
    kl = {"ispolnenie": dict(strogoe.ISPOLNENIE)}
    row = {"v": "1.0", "klyuch": kl, "period": "train_ce", "osnovnaya": True}
    with pytest.raises(ValueError):
        ledger_ce.Chitatel_ce().razobrat(dict(row, by_year={"2020": 1.0}))
    ledger_ce.Pisatel_ce(tmp_path, "a", "v").zapisat({"klyuch": kl, "period": "train_ce"})
    ledger_ce.Pisatel_ce(tmp_path, "b", "v").zapisat({"klyuch": kl, "period": "train_ce"})
    with pytest.raises(ValueError):
        ledger_ce.Chitatel_ce().chitat(tmp_path)
