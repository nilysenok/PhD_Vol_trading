"""Тестировщик PR #850: мутации в памяти (monkeypatch) на fastbt/metriki_diag.py, известный ответ руками."""
from __future__ import annotations

import sys
from math import isnan, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastbt import metriki_diag as D  # noqa: E402
from fastbt import metriki_e as E  # noqa: E402

R = [0.02, 0.0, 0.01, 0.0]
POZ = [1, 0, 1, 0]


def _seriya(n=504):
    daty = pd.bdate_range("2018-01-01", periods=n)
    i = np.arange(n)
    return daty, np.where(i % 2 == 0, 0.01, -0.005), np.where(i % 2 == 0, 1.0, 0.0)


def test_norm_pobajtno_ravno_metriki_e():
    assert D.sharpe_norm(R, POZ) == E.sharpe(R) / 0.5
    assert D.panel_diag(*_seriya()[:3], [], [], []).loc["итого", "Sharpe дн."] == E.sharpe(_seriya()[1])


def test_mutaciya_ddof0_v_sharpe_dn_lovitsya(monkeypatch):
    ok = D.sharpe_norm(R, POZ)
    r = np.asarray(R)
    monkeypatch.setattr(D, "sharpe_dn", lambda x: np.mean(x) * sqrt(252) / np.std(x, ddof=0))
    assert abs(D.sharpe_norm(R, POZ) - ok) > 0.5
    assert abs(D.sharpe_norm(R, POZ) - 2 * r.mean() * sqrt(252) / r.std(ddof=1)) > 0.5


def test_mutaciya_ekspoziciya_po_netto_lovitsya(monkeypatch):
    # шорт и лонг в разные дни: нетто [+1,-1,+1,-1] = 0 (ложный NaN), валовая = 1
    assert D.ekspoziciya([1, -1, 1, -1]) == 1.0
    assert D.prichina_norm([1, -1, 1, -1]) is None
    monkeypatch.setattr(D, "ekspoziciya", lambda p: abs(float(np.mean(p))))
    assert D.prichina_norm([1, -1, 1, -1]) is not None  # мутант даёт NaN — тест выше его бы уронил


def test_ekspoziciya_znakovaya_vhodnaya_dolya_zamechanie():
    # вход — одна знаковая доля на день: лонг 0,5 + шорт 0,5 одновременно нетто = 0 и валовой 1,0 не выразить
    assert D.ekspoziciya([0.5 - 0.5]) == 0.0


def test_stroki_nul_dnej_i_postoyanstvo_pnl():
    assert isnan(D.sharpe_sdelok([0.5, 0.5, 0.5], 1.0)) and D.prichina_sdelok([0.5, 0.5, 0.5], 1.0)
    assert not np.isinf(D.sharpe_sdelok([0.1] * 7, 1.0))
    assert isnan(D.sharpe_sdelok([1.0, 2.0], 0.0)) and D.prichina_sdelok([1.0, 2.0], 0.0)


def test_sdelki_okno_dva_s_polovinoi_goda():
    # 2,5 года: [3,-1,2,0,1] mean 1, sd(ddof=1)=√2.5; √(5/2.5)=√2 -> √2/√2.5
    v = D.sharpe_sdelok([3.0, -1.0, 2.0, 0.0, 1.0], 2.5)
    assert abs(v - sqrt(2) / sqrt(2.5)) < 1e-12
    assert D.sdelok_v_god(5, 2.5) == 2.0


def test_panel_itog_po_vsemu_oknu_a_ne_srednee_godovyh():
    daty, r, poz = _seriya()
    p = D.panel_diag(daty, r, poz, [2018, 2018, 2019, 2019], [2.0, 0.0, 1.0, -1.0], [1, 3, 2, 2])
    assert abs(p.loc["итого", "Sharpe дн."] - E.sharpe(r)) < 1e-12
    god = [p.loc[g, "Sharpe дн."] for g in (2018, 2019)]
    assert abs(p.loc["итого", "Sharpe дн."] - np.mean(god)) > 1e-6  # мутант «среднее годовых» отличим
    # N в год: итого 4/(504/252)=2; по годам: 2018 — 2 сделки / (дней2018/252)
    n18 = int((daty.year == 2018).sum())
    assert abs(p.loc[2018, "N сделок в год"] - 2 / (n18 / 252)) < 1e-12
    assert abs(p.loc["итого", "N сделок в год"] - 2.0) < 1e-12


def test_panel_sdelka_k_godu_vhoda_odno_pravilo():
    # сделка открыта в 2018, закрыта в 2019: god_sdelok = год входа (докстринг) -> считается в 2018
    daty, r, poz = _seriya()
    p = D.panel_diag(daty, r, poz, [2018, 2018, 2018], [1.0, 2.0, 3.0], [5, 5, 5])
    assert p.loc[2019, "N сделок в год"] == 0.0 and isnan(p.loc[2019, "Sharpe по сделкам (диагностика)"])
    assert p.loc[2019, "причина NaN"] and p.loc[2018, "N сделок в год"] > 0
    assert "год входа" in D.panel_diag.__doc__


def test_panel_cena_prostoev_i_podpisi():
    daty, r, poz = _seriya()
    p = D.panel_diag(daty, r, poz, [2018, 2018, 2019, 2019], [2.0, 0.0, 1.0, -1.0], [1, 3, 2, 2])
    razn = p["Sharpe норм. (при полной загрузке, диагностика)"] - p["Sharpe дн."]
    assert np.allclose(razn, p["цена простоев (диагностика)"], atol=1e-12)
    for c in p.columns:
        if c not in ("Sharpe дн.", "N сделок в год", "удержание, дн.", "причина NaN"):
            assert "диагностика" in c
