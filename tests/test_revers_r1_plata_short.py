"""Плата за шорт 5 % для A_zashch / TEST_C / D (scripts/revers/r1_a_plata.py): ответы известны, мутации ловятся, таблица сходится с a_cd.
Без сети и без кэша s9: чистые функции на синтетике + чтение опубликованных csv."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts" / "analysis", ROOT / "scripts" / "revers"):
    sys.path.insert(0, str(p))
import dve_metriki_a as D  # noqa: E402
import r1_a_plata as P  # noqa: E402

T = ROOT / "docs" / "tables" / "revers_r1"
# 2 даты × 3 пары; дата 0 — один день до следующей (cal = 1), дата 1 — выходные (cal = 3)
POS = np.array([-1.0, 1.0, 0.0, -0.5, -1.0, 1.0])
CAL = np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0])
DI = np.array([0, 0, 0, 1, 1, 1])
CNT = np.array([3.0, 3.0])
BAZA = np.array([1 * 1 / 365 / 3, (0.5 + 1.0) * 3 / 365 / 3])


def proverit(baza_fn, plata_fn) -> None:
    """Инварианты платы; на правильной реализации молчит, на мутанте — AssertionError."""
    assert np.allclose(baza_fn(POS, CAL, DI, CNT), BAZA, rtol=0, atol=1e-15)
    assert np.all(baza_fn(np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0]), CAL, DI, CNT) == 0), "плата на лонгах"
    x = np.array([0.01, -0.02])
    assert np.array_equal(plata_fn(x, BAZA, 0.0), x), "плата 0 меняет ряд"
    assert np.allclose(plata_fn(x, BAZA, 0.05), x - 0.05 * BAZA, rtol=0, atol=1e-15), "знак/размер платы"
    assert np.all(plata_fn(x, BAZA, 0.05) < x), "плата с неверным знаком"


def test_pravilnaya_realizaciya_prohodit():
    proverit(P.baza_shorta, P.x_s_platoy)


def test_mutatsiya_plata_na_longah_lovitsya():
    def bad(pos, cal, di, cnt):                                 # |поз| вместо max(0, −поз): лонги платят
        return np.bincount(di, np.abs(pos) * cal / 365.0, len(cnt)) / cnt
    with pytest.raises(AssertionError):
        proverit(bad, P.x_s_platoy)


def test_mutatsiya_tolko_longi_lovitsya():
    def bad(pos, cal, di, cnt):                                 # max(0, +поз): платят только лонги
        return np.bincount(di, np.maximum(0.0, pos) * cal / 365.0, len(cnt)) / cnt
    with pytest.raises(AssertionError):
        proverit(bad, P.x_s_platoy)


def test_mutatsiya_znak_platy_lovitsya():
    with pytest.raises(AssertionError):
        proverit(P.baza_shorta, lambda x, b, pl: x + pl * b)


def test_mutatsiya_plata_nol_ne_ravna_ishodnomu_lovitsya():
    with pytest.raises(AssertionError):
        proverit(P.baza_shorta, lambda x, b, pl: x - (pl + 0.01) * b)


def test_mutatsiya_bez_kalendarnyh_dnej_lovitsya():
    def bad(pos, cal, di, cnt):                                 # выходные не считаются (1 день вместо 3)
        return np.bincount(di, np.maximum(0.0, -pos) / 365.0, len(cnt)) / cnt
    with pytest.raises(AssertionError):
        proverit(bad, P.x_s_platoy)


def test_pustaya_lenta_i_nulevye_pozicii():
    z = np.zeros(4)
    assert np.all(P.baza_shorta(z, np.ones(4), np.array([0, 0, 1, 1]), np.array([2.0, 2.0])) == 0)
    assert P.baza_shorta(np.array([]), np.array([]), np.array([], int), np.array([1.0])).tolist() == [0.0]


def test_metriki_soglasovany_s_D_stroka_a_strogaya():
    """Тот же способ, что у якоря / A_strogaya: Sharpe и mSR совпадают с D.stroka(x, delo, None, plata, baza) до 1e-12."""
    rng = np.random.default_rng(7)
    x = rng.normal(0.0002, 0.01, 300)
    baza = np.where(rng.random(300) < 0.4, 0.3 / 365, 0.0)
    delo = np.ones(300, bool)
    for plata in (0.0, 0.05):
        r = D.stroka(x, delo, None, plata, baza)
        m = P.metriki(P.x_s_platoy(x, baza, plata), baza)
        assert abs(m["sharpe"] - r["sharpe"]) < 1e-12 and abs(m["msr"] - r["msr"]) < 1e-12
    assert P.metriki(x, baza)["dolya_short_dnej"] == pytest.approx(float(np.mean(baza > 0)))


def test_msr_vne_oblasti_nan_s_pometkoy():
    x = 0.01 + 0.0005 * np.arange(60) % 3                     # стабильно положительный ряд: mVaR ≤ 0
    m = P.metriki(x, np.zeros(60))
    assert np.isnan(m["msr"]) and "вне области" in m["primechanie"]


def test_stroki_obekta_chetyre_perioda_i_summa_po_godam():
    dates = np.arange("2022-01-03", "2024-12-31", dtype="datetime64[D]")
    x = np.random.default_rng(1).normal(0, 0.01, len(dates))
    baza = np.full(len(dates), 1 / 365)
    rows = P.stroki_obekta("A_zashch = A (s9)", x, baza, dates, "таблица")
    assert len(rows) == 8 and {r["period"] for r in rows} == {"полные", "2022", "2023", "2024"}
    full = [r for r in rows if r["plata_zaem"] == 0.05 and r["period"] == "полные"][0]
    assert full["dni"] == len(dates) and full["dolya_short_dnej"] == 1.0
    assert sum(r["dni"] for r in rows if r["plata_zaem"] == 0.05 and r["period"] != "полные") == len(dates)


@pytest.mark.parametrize("net", [False, True])
def test_opublikovannaya_tablica_plata0_ravna_a_cd(net):
    d = T / ("_net" if net else "")
    df = pd.read_csv(d / "a_plata_short5_test1.csv", comment="#")
    assert len(df) == 24 and set(df.obekt) == {o for _, o in P.OBJEKTY}
    assert set(df.plata_zaem) == {0.0, 0.05} and set(df.period) == {"полные", "2022", "2023", "2024"}
    assert P.sverka(df, d / "a_cd_test1.csv") <= 1e-12
    for o, g in df.groupby("obekt"):
        a, b = (g[(g.plata_zaem == pl) & (g.period == "полные")].iloc[0] for pl in (0.0, 0.05))
        assert b.sharpe < a.sharpe if a.dolya_short_dnej > 0 else b.sharpe == a.sharpe, o
