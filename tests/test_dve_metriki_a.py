"""Тесты scripts/analysis/dve_metriki_a.py::stroka — на синтетике с известным ответом и подложенными ошибками."""
import importlib.util
from pathlib import Path

import numpy as np

from fastbt import metriki_e as me

_p = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "dve_metriki_a.py"
_s = importlib.util.spec_from_file_location("dve_metriki_a", _p)
D = importlib.util.module_from_spec(_s)
_s.loader.exec_module(D)

RNG = np.random.default_rng(7)
X = RNG.normal(0.0004, 0.01, 400)
DELO = RNG.random(400) < 0.5
BAZA = np.abs(RNG.normal(0, 0.001, 400))
SH = lambda v: me.sr_dnevnoy(v) * np.sqrt(252)          # noqa: E731


def test_polnyj_ryad_i_razryv():
    r = D.stroka(X, DELO, None, 0.0, BAZA)
    assert np.isclose(r["sharpe"], SH(X)) and np.isclose(r["msr"], me.msr(X)) and np.isclose(r["razryv"], r["msr"] - r["sharpe"])
    assert np.isclose(r["skew"], me.asimmetriya(X)) and np.isclose(r["kurt"], me.ekscess(X))
    assert "sharpe_bez_fdr" not in r


def test_hvosty_vychyorkivayut_dni_nuzhnoj_storony():
    s = np.sort(X)
    assert np.isclose(D.stroka(X, DELO, "hud", 0.0, BAZA)["sharpe"], SH(s[10:]))
    assert np.isclose(D.stroka(X, DELO, "luch", 0.0, BAZA)["sharpe"], SH(s[:-10]))   # перепутанные стороны не пройдут


def test_v_dele_i_ekspoziciya():
    r = D.stroka(X, DELO, None, 0.0, BAZA)
    assert np.isclose(r["sharpe_v_dele"], SH(X[DELO])) and r["dni_v_dele"] == DELO.sum()
    assert np.isclose(r["ekspoziciya"], DELO.mean())


def test_plata_za_short_vychitaetsya():
    r = D.stroka(X, DELO, None, 0.05, BAZA)
    assert np.isclose(r["sharpe"], SH(X - 0.05 * BAZA))
    assert r["sharpe"] < D.stroka(X, DELO, None, 0.0, BAZA)["sharpe"]


def test_net_dnej_v_dele_ne_padaet():
    r = D.stroka(X, np.zeros(400, bool), None, 0.0, BAZA)
    assert r["dni_v_dele"] == 0 and r["sharpe_v_dele"] == 0.0 and r["ekspoziciya"] == 0.0
