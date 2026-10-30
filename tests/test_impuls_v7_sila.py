"""V7-5 (Ж-153 п. 1, 3): сила пробоя S10–S12 и сетка τ. Только синтетика.

Сила = (close − граница канала) / ATR таймфрейма; ATR по окну строго до бара решения (М-d1); знак — сторона.
Мутации: «ATR с баром решения» — test_sila_izvestnyj_otvet_vverh/vniz, test_sila_ne_zavisit_ot_diapazona_bara_resheniya;
«сила без знака стороны» — test_sila_izvestnyj_otvet_vniz, test_znak_sily_raven_storone;
τ=1 в основной сетке — test_tau_osnovnaya_setka_bez_edinicy.
"""
import numpy as np
import pandas as pd

from fastbt import bars_agg as ba
from fastbt import impuls_v7 as I
from fastbt.signals import v7_impuls as V
from test_impuls_v7 import b10_iz, sluchaynyj

DAY, POSLEDNIJ_1H = 8, range(42, 48)   # последний полный 1ч-бар дня к 18:30: слоты 42..47


def lenta(close_day=110.0, vverh=True, shirina=1.0, ndays=12):
    """Плоско 100 с размахом ±shirina; в день DAY на последнем полном 1ч-баре цена close_day."""
    b = b10_iz(lambda i, s: close_day if (i == DAY and s in POSLEDNIJ_1H) else 100.0, ndays)
    b["high"], b["low"] = b["close"] + shirina, b["close"] - shirina
    return b


def znach(b, tf="1h"):
    s = I.sila_proboja(b, tf, "18:30")
    return float(s.loc[pd.bdate_range("2016-01-04", periods=DAY + 1)[-1]])


def test_sila_izvestnyj_otvet_vverh():
    # граница = max high 20 баров = 101, ATR = 2 (TR плоских баров = 2), close = 110 → (110 − 101) / 2
    assert znach(lenta(110.0)) == 4.5


def test_sila_izvestnyj_otvet_vniz():
    # граница = min low = 99, close = 90 → (90 − 99) / 2 = −4.5 (знак стороны)
    assert znach(lenta(90.0)) == -4.5


def test_sila_ne_zavisit_ot_diapazona_bara_resheniya():
    a, b = lenta(110.0), lenta(110.0)
    m = (b["date"] == pd.bdate_range("2016-01-04", periods=DAY + 1)[-1]) & b["slot"].isin(POSLEDNIJ_1H)
    b.loc[m, "high"] += 50.0
    b.loc[m, "low"] -= 30.0          # ATR «с баром решения» изменился бы
    assert znach(a) == znach(b) == 4.5


def test_znak_sily_raven_storone():
    b = sluchaynyj(120, seed=3)
    for tf in ("1h", "2h"):
        sila = I.sila_proboja(b, tf, "18:30")
        side = I.proboj(b, tf, "18:30")
        assert (np.sign(sila).astype(int) == side.astype(int)).all()
        assert (sila[side != 0].abs() > 0).all()


def test_signal_dnya_prezhnij_i_strength_modul():
    b = sluchaynyj(120, seed=4)
    for sid in ("S10", "S11", "S12"):
        sila = I.sila_dnya(b, sid, "18:30")
        side = I.signal_dnya(b, sid, "18:30")
        assert (np.sign(sila).astype(int) == side.reindex(sila.index).astype(int)).all()


def test_tau_osnovnaya_setka_bez_edinicy():
    assert V.TAU == (2, 3, 5) and V.TAU_SPRAVKA == (1,)
    assert all(p["tau"] != 1 for p in V.setka())
    assert {p["tau"] for p in V.setka()} == {2, 3, 5}
    sp = V.setka(spravka=True)
    assert {p["tau"] for p in sp if p.get("rol") == "spravka"} == {1}
    assert all(p.get("rol") != "spravka" for p in V.setka() if p["tau"] != 1)
    assert len(V.setka()) == 3 * 2 * 3
