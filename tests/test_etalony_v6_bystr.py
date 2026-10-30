"""Быстрый слой v6 = эталонный путь внутри набора каждой машины (дирижёр, доработка #722): одинаковый сбой на Маке и сервере не проходит."""
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/server"))
import ekviv_io  # noqa: E402
import ekviv_sravn as es  # noqa: E402
import proverka_ekviv as pe  # noqa: E402

META = dict(kod="k1", vhody={"S1": "v1"}, kub={"S1": "c1"}, daty={"S1": "d1"})


def _zapis(sharpe=1.5):
    return dict(sig="{}", disk='{"status": "ok", "n": 7, "sharpe": null}', hex={"/sharpe": float(sharpe).hex()},
                p=np.array([0.0, 0.01, -0.02]), t=np.array([0.1, 0.2]), pz=np.array([0.0, 1.0, 1.0]),
                sd=np.array([0, 2, 0], np.int64), vi=np.array([[0, 1, 1], [0, 3, -1]], np.int64))


def _nabor():
    return {"yakor__S1": _zapis(), "tochka__S1__5": _zapis(2.0), "v6yakor__S1": _zapis(), "v6tochka__S1__7": _zapis(3.0),
            "v6bystr__S1__7": _zapis(3.0)}


def _isporchen(rez, dx=1e-9):
    rez = copy.deepcopy(rez)
    h = rez["v6bystr__S1__7"]["hex"]
    h["/sharpe"] = (float.fromhex(h["/sharpe"]) + dx).hex()
    return rez


def test_odinakovye_para_ok():
    assert es.bystr_protiv_etalona(_nabor()) == []


def test_mutaciya_bystr_1e9_na_obeih_mashinah_odinakovo_lovitsya():
    mak = _isporchen(_nabor())
    server = copy.deepcopy(mak)
    assert es.sravnit(mak, META, server, copy.deepcopy(META), 1e-12)["kod"] == 0     # Мак ↔ сервер этого сбоя не видит (пробел)
    for rez in (mak, server):
        bad = es.bystr_protiv_etalona(rez)
        assert [b[:2] for b in bad] == [("v6bystr__S1__7", 3)]


def test_diskretnoe_rashozhdenie_bystr_kod_1():
    rez = _nabor()
    rez["v6bystr__S1__7"]["sd"] = np.array([0, 1, 0], np.int64)
    assert es.bystr_protiv_etalona(rez)[0][1] == 1


def test_net_parnoj_zapisi_kod_1():
    rez = _nabor()
    del rez["v6tochka__S1__7"]
    assert es.bystr_protiv_etalona(rez)[0][1] == 1


def test_proverka_ekviv_main_fail_pri_odinakovom_sboe(tmp_path):
    def run(rez):
        f = str(tmp_path / "e.npz")
        ekviv_io.sohranit(f, rez, META)
        return pe.main(["--etalony", f, "--moi", f, "--otmetka", str(tmp_path / "p" / "OK"), "--kod-sha", "a", "--dannye-sha", "b"])
    assert run(_nabor()) == 0
    assert run(_isporchen(_nabor())) == 3
    assert not (tmp_path / "p" / "OK").exists()                  # отметки OK нет — счёт не стартует
