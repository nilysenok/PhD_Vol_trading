"""Тестировщик PR #875: мутации к golaya_otnositelno_v7 (комиссия на конце окна, k2 двигает выходы, Д4 начала ряда, схема schet с флагами)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "analysis"), str(ROOT / "scripts" / "server")]
import golaya_otnositelno_v7 as go  # noqa: E402
from test_golaya_otnositelno_v7 import D  # noqa: E402,F401  (фикстура)


def _derzhat_do_konca(D):
    S = np.zeros(D["c"].shape, np.int8)
    S[300, 0] = 1                                              # без флагов выхода — позиция до последнего close окна
    return S


def _kom_posledniy_den(D, schet):
    o = schet(D, _derzhat_do_konca(D))
    return o["kom"][np.flatnonzero(o["kom"])[-1]], np.flatnonzero(o["kom"])[-1]


def test_komissiya_vyhoda_na_konce_okna_platitsya(D):
    kom, den = _kom_posledniy_den(D, go.schet)
    assert den == np.flatnonzero(D["o"].iloc[:, 0].notna().to_numpy())[-1] and kom > 0   # выход платится в последний день с ценой


def test_mutaciya_kom_vyhoda_na_konce_okna_lovitsya(D, monkeypatch):
    orig = go.ispolnit

    def bez_vyhoda(*a, **k):
        r = orig(*a, **k)
        i = np.flatnonzero(r["kom"])[-1]
        r["kom"][i] = 0.0
        return r
    monkeypatch.setattr(go, "ispolnit", bez_vyhoda)
    with pytest.raises(AssertionError):
        test_komissiya_vyhoda_na_konce_okna_platitsya(D)


def _k2_ne_dvigaet_vyhody(signaly, D):
    base = signaly(D, "S1", "sektor")
    for ctl in (("k2", 100), ("k2", 77), ("placebo", 7)):
        c = signaly(D, "S1", "sektor", ctl)
        assert np.array_equal(base[1], c[1]) and np.array_equal(base[2], c[2]), ctl
    assert base[1].any() or base[2].any()


def test_k2_i_plasebo_ne_dvigayut_vyhody(D):
    _k2_ne_dvigaet_vyhody(go.signaly, D)


def test_mutaciya_k2_dvigaet_i_vyhody_lovitsya(D):
    def mutant(D, sid, rezhim="net", ctl=None):
        S, EL, ES = go.signaly(D, sid, rezhim, ctl)
        return S, (np.roll(EL, 100, axis=0) if ctl else EL), (np.roll(ES, 100, axis=0) if ctl else ES)
    with pytest.raises(AssertionError):
        _k2_ne_dvigaet_vyhody(mutant, D)


def test_k2_lag0_dayot_ishodnye_summy_dnevnyh_ryadov(D):
    a = go.schet(D, *go.signaly(D, "S2", "imoex"))
    b = go.schet(D, *go.signaly(D, "S2", "imoex", ("k2", 0)))
    assert all(np.array_equal(a[k], b[k]) for k in ("pnl", "kom", "zaem", "stavka"))


def test_nachalo_d4_posle_poslednego_dnya_bez_fona(D):
    td = D["td"][D["tk"][0]][0]
    fon = np.full(td.n, 0.001)
    fon[0] = np.nan                                            # первый день — не используется
    assert go._nachalo(td, fon) == 0
    fon[[50, 120]] = np.nan
    assert go._nachalo(td, fon) == 120                         # ряд стартует с последнего дня-дыры, дыра не заполняется


def test_d4_signaly_do_nachala_ryada_nulevye(D, monkeypatch):
    td, ix = D["td"][D["tk"][0]]
    fon_real = go._fon(D, "sektor")[0]
    f2 = fon_real.copy()
    f2.loc[pd.DatetimeIndex(td.dates)[150], D["tk"][0]] = np.nan
    D2 = dict(D, _sig={}, _fon={"sektor": (f2, go._fon(D, "sektor")[1])}, obrez={})
    S = go.signaly(D2, "S1", "sektor")[0]
    assert D2["obrez"][("sektor", D["tk"][0])] == 150
    assert not S[ix[:150], 0].any()
