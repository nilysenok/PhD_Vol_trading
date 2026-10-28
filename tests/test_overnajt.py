"""Овернайт-разложение (Ж-103 п. 2): известные ответы на синтетике, без данных. Мутация: ночь и день перепутаны (test_mutaciya_chasti_pereputany_lovitsya).
Знаки шорта, дивиденд в ночь, пустая лента, пауза, нулевая доходность и дисперсия, close = open / close = следующий open у книги S7–S9."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
import overnajt_lib as L  # noqa: E402

from fastbt.exec_e_types import Trade  # noqa: E402

O = np.array([100.0, 102.0, 101.0, 105.0, 104.0])
C = np.array([101.0, 100.0, 103.0, 104.0, 104.5])
IX, N = [0, 1, 2, 3, 4], 5
W = np.ones(N)


def sdelka(e, x, ex, px, side=1, entry_px=None):
    return Trade("T", e, e, 0, O[e] if entry_px is None else entry_px, exit_day=x, exit_bar=0, exit_px=px, reason="vremya", side=side)


def raz(tr, div=None, nal=0.87, fn=L.razlozhit_sdelki):
    return fn([tr], W, O, C, div, nal, IX, N)


def odna_noch_ok(fn) -> bool:
    """Вход по close дня 1, выход по open дня 2: внутридневной части нет, ночь = ln(open2/close1) (строка 1)."""
    I, Nn = raz(Trade("T", 1, 1, 0, C[1], exit_day=2, exit_bar=0, exit_px=O[2], reason="vremya"), fn=fn)
    return bool(np.allclose(I, 0.0) and Nn[1] == np.log(O[2] / C[1]) and np.count_nonzero(Nn) == 1)


def test_odna_noch_vsya_dohodnost_v_overnajt():
    assert odna_noch_ok(L.razlozhit_sdelki)


def test_odin_den_vsya_dohodnost_vnutridnevnaya():
    """Вход по open дня 2, выход по close того же дня: ночи нет."""
    I, Nn = raz(sdelka(2, 2, None, C[2]))
    assert np.allclose(Nn, 0.0) and I[1] == np.log(C[2] / O[2])


def test_summa_chastey_ravna_logu_ceny_vyhoda_k_vhodu():
    """Телескопирование: за всю сделку I + N = ln(выход/вход) (без дивидендов); шорт — с обратным знаком."""
    for side in (1, -1):
        I, Nn = raz(sdelka(1, 3, None, 104.0, side))
        assert abs((I + Nn).sum() - side * np.log(104.0 / O[1])) < 1e-12


def test_pyatidnevnaya_raskladka_po_dnyam():
    I, Nn = raz(sdelka(1, 3, None, O[3]))                      # вход open1, выход open3 (по сигналу, open2-режим)
    assert np.allclose(I[[0, 1]], [np.log(C[1] / O[1]), np.log(C[2] / O[2])]) and I[2] == 0.0
    assert np.allclose(Nn[[1, 2]], [np.log(O[2] / C[1]), np.log(O[3] / C[2])])


def test_dividend_idyot_v_noch_i_shorty_platyat():
    base = raz(sdelka(1, 3, None, O[3]))
    dv = np.array([0, 0, 2.0, 0, 0])
    I, Nn = raz(sdelka(1, 3, None, O[3]), div=dv)
    assert np.allclose(I, base[0]) and np.isclose(Nn[1] - base[1][1], np.log((C[2] + 2 * 0.87) / C[2]))
    Is, Ns = raz(sdelka(1, 3, None, O[3], -1), div=dv)                    # шорт платит дивиденд: знак обратный, в ночь
    assert np.allclose(Is, -I) and np.allclose(Ns, -Nn)


def test_pustaya_lenta_i_dyra():
    I, Nn = L.razlozhit_sdelki([], W, O, C, None, 0.87, IX, N)
    assert not I.any() and not Nn.any()
    I, Nn = L.razlozhit_sdelki([sdelka(1, 3, None, O[3])], W, O, C, None, 0.87, [0, 1, 3, 4], N)   # день 2 — пауза: ночь идёт от close 1
    assert np.isclose(Nn[1], np.log(O[3] / C[1])) and I[2] == 0.0


def test_pustoy_ryad_dolei_ne_vydumyvaet():
    d = L.dolya_noch(np.zeros(10), np.zeros(10))
    assert np.isnan(d["dolya_disp_noch"]) and np.isnan(d["dolya_doh_noch"])
    assert L.flag_p2(d["dolya_disp_noch"], d["dolya_doh_noch"]) == "н/д"


def test_dolya_dispersii_i_flag():
    rng = np.random.default_rng(1)
    I, Nn = rng.normal(0.004, 0.002, 1000), rng.normal(0.0, 0.01, 1000)        # ночь: шум, доходность ~0
    k = L.dolya_noch(I, Nn)
    assert k["dolya_disp_noch"] > 0.9 and k["dolya_doh_noch"] < 0.2 and L.flag_p2(k["dolya_disp_noch"], k["dolya_doh_noch"]) == "да"
    k2 = L.dolya_noch(Nn, I)                                                    # перепутанные части: флага нет
    assert L.flag_p2(k2["dolya_disp_noch"], k2["dolya_doh_noch"]) == "нет"
    assert abs(L.dolya_noch(I, Nn)["dolya_disp_noch"] + L.dolya_noch(Nn, I)["dolya_disp_noch"] - 1) < 1e-9   # аддитивность


def test_flag_granitsy():
    assert L.flag_p2(0.5, 0.1) == "нет" and L.flag_p2(0.51, 0.2) == "нет" and L.flag_p2(0.51, 0.19) == "да"


def test_sharpe_v_dele_otlichaetsya_ot_obychnogo():
    x = np.array([0.01, -0.01, 0.0, 0.0, 0.02, 0.0])
    r = L.stroka("k", "S", x, x * 0, x, x, x != 0)
    assert r["sharpe_v_dele"] != r["sharpe"] and r["dolya_dnej_v_dele"] == 0.5


def test_kniga_razlozhenie_izvestnye_otvety():
    from fastbt import kross_seksiya as ks
    rng = np.random.default_rng(2)
    T, n = 60, 4
    P, Dv = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (T, n)), 0)), np.zeros((T, n))
    Dv[30, 1] = 1.5
    Wm = np.zeros((T, n))
    Wm[::3] = [0.25, -0.25, 0.25, -0.25]
    o = ks.kniga(Wm, P, Dv, np.arange(T).astype("datetime64[D]"), np.zeros(T), 5, 5, 3)
    I, Nn = L.kniga_razlozhenie(Wm, P, P * 1.001, Dv, 5, 5, 3)
    assert np.allclose(I + Nn, o["pnl"], atol=1e-12) and abs(Nn).sum() > 0
    I0, N0 = L.kniga_razlozhenie(Wm, P, P, Dv, 5, 5, 3)                          # close = open: вся доходность в ночь
    assert not I0.any() and np.allclose(N0, o["pnl"], atol=1e-12)
    Pn = np.vstack([P[1:], P[-1:]])                                              # close = следующий open (D = 0): ночи нет
    I1, N1 = L.kniga_razlozhenie(Wm, P, Pn, np.zeros((T, n)), 5, 5, 3)
    assert np.allclose(N1, 0.0, atol=1e-12)
    Ic, Nc = L.kniga_razlozhenie(Wm, P, np.full((T, n), np.nan), Dv, 5, 5, 3)    # нет close: всё в ночь, не NaN
    assert not Ic.any() and np.isfinite(Nc).all()


def peremeshannye(tr_fn):
    """Мутант: части поменяны местами."""
    def f(*a, **k):
        I, Nn = tr_fn(*a, **k)
        return Nn, I
    return f


def test_mutaciya_chasti_pereputany_lovitsya():
    assert not odna_noch_ok(peremeshannye(L.razlozhit_sdelki))     # мутант отдаёт ночь в день: проверка «одна ночь» его ловит
    assert odna_noch_ok(L.razlozhit_sdelki)
