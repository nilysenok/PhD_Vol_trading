"""V6-12b, синтетика (данные не читает): GOTOVY по осям (canon_v6) и единицы fastbt/eval_v6_osi.py.
Гейты: GOTOVY открывает только подключённые оси (Соболь не варьирует неподключённую; мутация «лишняя ось в GOTOVY» ловится);
fkey содержит новые оси; тень паузы: окно без строки d, нет полного окна — вход разрешён. Реальные данные — test_eval_v6b_real.py."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fastbt import canon_v5, canon_v6, eval_v6, eval_v6_osi as osi6
from fastbt.eval_v6 import fkey, razdelit
from fastbt.etap1_eval_v2 import NeSchitana

SIDS, B_OSI = canon_v5.SIDS, osi6.DVIG + osi6.PAUZA


# ---------------- GOTOVY: только подключённые оси ----------------
def test_gotovo_po_osyam():
    g = canon_v6.gotovo
    assert g(["V6-1a", "V6-12"], None, "x") and not g(["V6-1a", "V6-12"], {"V6-1a"}, "x")
    assert not g(["V6-12"], {"V6-12:y"}, "x") and g(["V6-12"], {"V6-12:x"}, "x") and g(["V6-12"], {"V6-12"}, "x")
    assert not g(["V6-12:x"], {"V6-12"}, None) and g([], set(), "x")


def test_v12_ravno_podklyucheno():
    assert set(canon_v6.V12) == set(eval_v6.PODKLYUCHENO) - {"vhod_tip"}
    assert set(B_OSI) <= set(canon_v6.V12)


@pytest.mark.parametrize("sid", SIDS)
def test_sobol_ne_variruet_nepodklyuchennuyu_os(sid):
    kub = canon_v6.load_kub(sid)
    nov, ras = canon_v6.novye(kub), canon_v6.rasshireno(kub)
    zakr = set(nov) - set(eval_v6.PODKLYUCHENO)
    pts = [c for _, c in canon_v6.tochki_v6(kub, 0, 1500)[0]]
    assert set().union(*[set(c) for c in pts]) & zakr == set()
    assert not any(str(c[n]) in map(str, r["v6_dobavleno"]) for c in pts for n, r in ras.items() if n in c)
    assert {n for n in B_OSI if n in nov} <= set().union(*[set(c) for c in pts]), "подключённая ось ни разу не выпала"
    for c in pts[:400]:
        razdelit(sid, c)                                          # ни одна точка Соболя не даёт «ещё не подключена»


def test_mutaciya_gotovy_s_lishnej_os_lovitsya():
    kub = canon_v6.load_kub("S1")
    pts = [c for _, c in canon_v6.tochki_v6(kub, 0, 800, gotovy=canon_v6.GOTOVY | {"V6-12:tf"})[0]]
    plohie = [c for c in pts if "tf" in c]
    assert plohie
    with pytest.raises(ValueError, match="ещё не подключена"):
        razdelit("S1", plohie[0])


# ---------------- ключ кэша ----------------
def test_fkey_soderzhit_novye_osi():
    ks = {fkey({}, {k: v}) for k, v in (("time_decay", "da"), ("sigma_ist", "rv5"), ("sigma_ist", "rv66"), ("ed_lim", "atr14"),
                                         ("zhizn", 12), ("zhizn", 36), ("pauza_porog", 0.5), ("pauza_porog", 0.0))}
    assert len(ks) == 8 and fkey({}, {"pauza_porog": 0.5}) != fkey({}, {"pauza_porog": 0.5, "pauza_okno": 42})


# ---------------- единицы eval_v6_osi ----------------
def _r(tau="net", tejk="net"):
    return SimpleNamespace(pt4={"tau": tau, "sigma": "rv22"}, nov={"tejk_signal": tejk, "zhizn": "den"}, ext={})


def test_primenit_i_proverit():
    r = _r(10)
    osi6.primenit(r, {"time_decay": "da", "sigma_ist": "rv5", "zhizn": 12}, {"rv5": 1})
    assert r.ext == {"time_decay": True} and r.pt4["sigma"] == "rv5" and r.nov["zhizn"] == 12
    osi6.proverit(r, {"time_decay": "da"})
    for bad in (_r("net"), _r(10, "rodnoy")):
        with pytest.raises(NeSchitana, match="time_decay"):
            osi6.proverit(bad, {"time_decay": "da"})
    osi6.proverit(_r("net"), {})                                   # без оси — без проверки
    with pytest.raises(ValueError, match="sigma_ist"):
        osi6.primenit(_r(10), {"sigma_ist": "rv5"}, {"rv22": 1})


def _ten(vals, n=200):
    ix = pd.bdate_range("2018-01-02", periods=n)
    return {"T": pd.Series(vals, index=ix)}, ix


def test_razreshit_pauza_po_sharpu_teni():
    rng = np.random.default_rng(1)
    ten, ix = _ten(np.r_[rng.normal(0.002, 0.01, 100), rng.normal(-0.004, 0.01, 100)])
    a = osi6.razreshit({"pauza_porog": 0.0, "pauza_okno": 42}, "T", ix, ten)
    assert a[:42].all() and a[60:100].mean() > 0.6 and not a[150:].any()          # нет полного окна → вход разрешён; потом пауза
    assert osi6.razreshit({"pauza_porog": -99.0}, "T", ix, ten).all()              # порог ниже любого Шарпа → паузы нет
    lag = osi6.razreshit({"pauza_porog": 0.0, "pauza_okno": 42}, "T", ix, {"T": ten["T"].shift(1)})
    assert not np.array_equal(a, lag)


def test_razreshit_den_d_ne_v_okne():
    ten, ix = _ten(np.r_[np.full(120, 0.001), [-9.0], np.full(79, 0.001)])
    ten["T"] += np.random.default_rng(2).normal(0, 1e-4, 200)
    a = osi6.razreshit({"pauza_porog": 0.0, "pauza_okno": 63}, "T", ix, ten)
    assert a[120] and not a[121]                                                    # строка d видна только со следующего дня
