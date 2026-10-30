"""Диагностические метрики Ж-148 (fastbt/metriki_diag.py): известный ответ руками, мутации в памяти.
Мутации: ddof=0 (№ 12), экспозиция только по дням с позицией, √N вместо √(N/лет)."""
from __future__ import annotations

import inspect
import sys
from math import isnan, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastbt import metriki_diag as D  # noqa: E402
from fastbt import metriki_e as E  # noqa: E402

R = [0.02, 0.0, 0.01, 0.0]


def sd_dn_rukami(r, ddof=1):
    r = np.asarray(r, float)
    return r.mean() * sqrt(252) / r.std(ddof=ddof)


def test_sharpe_dn_imya_iz_metriki_e():
    assert D.sharpe_dn is E.sharpe  # импорт, не копия


def test_ekspoziciya_valovaya_i_dni_bez_pozicii():
    assert D.ekspoziciya([1, 0, 1, 0]) == 0.5
    assert D.ekspoziciya([-1, 0.5, 0, 0]) == 0.375  # |шорт| + |лонг|, 4 дня
    assert D.ekspoziciya([0, 0, 0]) == 0.0


def test_ekspoziciya_portfel_valovaya_po_tikeram():
    w = np.array([[0.5, -0.5], [0.0, 0.0]])  # день 1: лонг A 0,5 и шорт B 0,5
    assert D.ekspoziciya(w) == 0.5  # (1,0 + 0)/2
    assert D.ekspoziciya(w[:1]) == 1.0
    assert D.ekspoziciya(pd.DataFrame(w[:1], columns=["A", "B"])) == 1.0
    assert D.ekspoziciya([-0.5, 0.5]) == 0.5  # 1-D: один тикер, |w|
    netto = abs(w[0].sum())  # мутация |Σw| вместо Σ|w| даёт 0
    assert D.ekspoziciya(w[:1]) != netto


def test_sharpe_norm_i_panel_s_matricei():
    w = np.array([[0.5, -0.5], [0.0, 0.0]] * 2)
    assert abs(D.sharpe_norm(R, w) - 2 * sd_dn_rukami(R)) < 1e-9
    daty = pd.bdate_range("2018-01-01", periods=4)
    p = D.panel_diag(daty, R, w, [2018, 2018], [2.0, 0.0], [1, 1])
    assert abs(p.loc["итого", D.K_NORM] - 2 * sd_dn_rukami(R)) < 1e-9


def test_norm_pri_polovine_dnej_v_pozicii_dvazhdy_dnevnoy():
    n = D.sharpe_norm(R, [1, 0, 1, 0])
    assert abs(n - 2 * sd_dn_rukami(R)) < 1e-9
    assert abs(sd_dn_rukami(R) - 12.4354) < 1e-3  # ответ руками: 0,0075·√252/0,0095743


def test_mutaciya_ekspoziciya_tolko_po_dnyam_s_poziciei_lovitsya():
    mut = np.mean([1, 1])  # среднее только по дням с позицией = 1
    assert D.ekspoziciya([1, 0, 1, 0]) != mut
    assert abs(D.sharpe_norm(R, [1, 0, 1, 0]) - sd_dn_rukami(R) / mut) > 1.0


def test_mutaciya_ddof0_lovitsya():
    assert abs(D.sharpe_norm(R, [1, 0, 1, 0]) - 2 * sd_dn_rukami(R, ddof=0)) > 0.5
    # сделочный: [2, 0] за 1 год: 1/√2·√2 = 1 при ddof=1; при ddof=0 было бы √2
    assert abs(D.sharpe_sdelok([2.0, 0.0], 1.0) - 1.0) < 1e-12


def test_dve_sdelki_znak():
    # +3 и −1: mean 1, sd(ddof=1)=√8; N=2, 2 года → √1
    v = D.sharpe_sdelok([3.0, -1.0], 2.0)
    assert abs(v - 1.0 / sqrt(8) * 1.0) < 1e-12
    assert D.sharpe_sdelok([-3.0, 1.0], 2.0) < 0  # перепутанный знак


def test_mutaciya_koren_N_bez_let_lovitsya():
    mut = 1.0 / sqrt(8) * sqrt(2)  # √N вместо √(N/лет)
    assert abs(D.sharpe_sdelok([3.0, -1.0], 2.0) - mut) > 0.1


def test_odna_sdelka_nan_s_prichinoi():
    assert isnan(D.sharpe_sdelok([1.0], 1.0))
    assert D.prichina_sdelok([1.0], 1.0)
    assert isnan(D.sharpe_sdelok([], 1.0))
    assert D.prichina_sdelok([1.0, 2.0], 1.0) is None


def test_ekspoziciya_nol_nan_s_prichinoi():
    assert isnan(D.sharpe_norm(R, [0, 0, 0, 0]))
    assert D.prichina_norm([0, 0, 0, 0])
    assert D.prichina_norm([1, 0]) is None


def test_cena_prostoev_i_sdelok_v_god_i_uderzhanie():
    assert abs(D.cena_prostoev(R, [1, 0, 1, 0]) - sd_dn_rukami(R)) < 1e-9
    assert D.sdelok_v_god(6, 2.0) == 3.0
    assert D.uderzhanie_srednee([1, 3]) == 2.0
    assert isnan(D.uderzhanie_srednee([]))


def test_kontrakt_argumenta_posle_komissii():
    assert list(inspect.signature(D.sharpe_sdelok).parameters) == ["pnl_sdelok_izb", "let_okna"]
    doc = D.sharpe_sdelok.__doc__
    assert "после комиссии" in doc and "golaya_v7_isp" in doc


def test_panel_podpisi_i_stroki():
    daty = pd.bdate_range("2018-01-01", periods=504)
    r = np.where(np.arange(504) % 2 == 0, 0.01, -0.005)
    poz = np.where(np.arange(504) % 2 == 0, 1.0, 0.0)
    p = D.panel_diag(daty, r, poz, [2018, 2018, 2019, 2019], [2.0, 0.0, 1.0, -1.0], [1, 3, 2, 2])
    assert "Sharpe норм. (при полной загрузке, диагностика)" in p.columns
    assert "Sharpe по сделкам (диагностика)" in p.columns
    assert "цена простоев (диагностика)" in p.columns
    assert "N сделок в год" in p.columns and "удержание, дн." in p.columns
    assert list(p.index)[-1] == "итого" and 2018 in p.index and 2019 in p.index
    assert p.loc[2018, "удержание, дн."] == 2.0
    assert p.loc["итого", "N сделок в год"] == 4 / (504 / 252)
