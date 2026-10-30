"""Лист V7-2: маска входа k_okup (Ж-153). Известный ответ, M-d1, мутация «доля от нетто», выкл., < 20 сделок."""
import numpy as np
import pandas as pd

from fastbt import k_okup_v7 as K

D = pd.Timestamp("2018-09-14")  # день входа


def _tr(n, tercil, hold, gross, kom, start=0, vyhod0="2018-01-02"):
    dni = pd.bdate_range(vyhod0, periods=start + n)[start:]
    return pd.DataFrame({"vyhod": dni, "tercil": tercil, "uderzhanie": hold, "gross": gross, "kom": kom, "short": 0.0})


def _zhurnal():
    # терциль 1: 10 сделок удержание 1, 10 — удержание 5; терциль 2: 20 сделок удержание 1
    return pd.concat([_tr(10, 1, 1, 0.5, 0.1), _tr(10, 1, 5, 2.0, 0.2, 10), _tr(20, 2, 1, 0.5, 0.2, 20)], ignore_index=True)


def _vh(t):
    return pd.DataFrame({"vhod": [D], "tercil": [t]})


def test_izvestnyj_otvet_ozhidanie():
    # веса удержаний по всем сделкам окна: корзина «1» 30/40, «4–10» 10/40
    r = K.maska_vhoda(_zhurnal(), _vh(1), 2).iloc[0]
    assert np.isclose(r.pnl_hat, 0.75 * 0.5 + 0.25 * 2.0) and np.isclose(r.izd_hat, 0.75 * 0.1 + 0.25 * 0.2)
    assert r.n == 20 and not r.malo and r.razresheno
    # терциль 2: ячейка только «1» — веса перенормированы по ячейкам с данными; 0,5 ≥ 2·0,2, но 0,5 < 3·0,2
    r2, r3 = K.maska_vhoda(_zhurnal(), _vh(2), 2).iloc[0], K.maska_vhoda(_zhurnal(), _vh(2), 3).iloc[0]
    assert r2.razresheno and not r3.razresheno and np.isclose(r2.pnl_hat, 0.5)


def test_mutaciya_dolya_ot_netto_lovitsya():
    # канон: валовой ≥ k·издержки; от нетто (валовой − издержки ≥ k·издержки) блокировал бы вход при k=2 (0,3 < 0,4)
    assert K.maska_vhoda(_zhurnal(), _vh(2), 2).razresheno.iloc[0]
    assert not K.maska_vhoda(_zhurnal(), _vh(2), 2, baza="netto").razresheno.iloc[0]


def test_md1_sdelka_zakrytaya_v_d_ne_vhodit():
    z = pd.concat([_tr(19, 3, 1, 0.5, 0.1), _tr(1, 3, 1, 9.0, 0.1, 19, "2018-01-02")], ignore_index=True)
    z.loc[19, "vyhod"] = D                                    # закрыта в день d: не должна учитываться
    r = K.maska_vhoda(z, _vh(3), 2).iloc[0]
    assert r.n == 19 and r.malo and np.isclose(r.pnl_hat, 0.5)
    z.loc[19, "vyhod"] = D - pd.offsets.BDay(1)               # закрыта в d−1: учитывается
    r = K.maska_vhoda(z, _vh(3), 2).iloc[0]
    assert r.n == 20 and not r.malo and r.pnl_hat > 0.5


def test_vykl_vse_razresheno_i_net_flagov():
    z = _zhurnal().assign(gross=-5.0)                         # даже убыточные ячейки
    r = K.maska_vhoda(z, pd.DataFrame({"vhod": [D, D], "tercil": [1, 3]}), None)
    assert r.razresheno.all() and not r.malo.any()


def test_menshe_20_razresheno_s_flagom():
    z = _tr(19, 1, 1, -1.0, 0.5)                              # убыточно, но сделок < 20
    r = K.maska_vhoda(z, _vh(1), 3).iloc[0]
    assert r.razresheno and r.malo and r.n == 19
    r = K.maska_vhoda(z.iloc[:0], _vh(1), 3).iloc[0]          # пустой журнал
    assert r.razresheno and r.malo and r.n == 0


def test_okno_252_otsekaet_starye():
    z = _tr(20, 1, 1, 0.5, 0.1, 0, "2016-01-04")              # закрыты задолго до окна
    r = K.maska_vhoda(z, _vh(1), 2).iloc[0]
    assert r.n == 0 and r.malo
