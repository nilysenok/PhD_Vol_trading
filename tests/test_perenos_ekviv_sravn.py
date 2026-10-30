"""Перенос на сервер: сравнение эталона в два уровня (дискретное строго, непрерывное с допуском DOPUSK) — на синтетике."""
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/server"))
import ekviv_io  # noqa: E402
import ekviv_sravn as es  # noqa: E402

META = dict(kod="k1", vhody={"S1": "v1"}, kub={"S1": "c1"}, daty={"S1": "d1"})


def _zapis(sharpe=1.5):
    return dict(sig="{}", disk='{"status": "ok", "n": 7, "sharpe": null}', hex={"/sharpe": float(sharpe).hex()},
                p=np.array([0.0, 0.01, -0.02]), t=np.array([0.1, 0.2]), pz=np.array([0.0, 1.0, 1.0]),
                sd=np.array([0, 2, 0], np.int64), vi=np.array([[0, 1, 1], [0, 3, -1]], np.int64))


def _nabor():
    return {"yakor__S1": _zapis(), "tochka__S1__5": _zapis(2.0)}


def _sr(moi, dopusk=0.0, meta=None):
    return es.sravnit(_nabor(), META, moi, meta or copy.deepcopy(META), dopusk)


def test_odinakovye_ok_i_max_delta_nol():
    r = _sr(_nabor())
    assert r["kod"] == 0 and r["max_yakor"] == 0.0 and r["max_tochki"] == 0.0


def test_odin_ulp_v_tochke_pri_dopuske_nol_fail_tolko_nepreryvnogo():
    moi = _nabor()
    moi["tochka__S1__5"]["hex"]["/sharpe"] = float(np.nextafter(2.0, 3.0)).hex()
    r = _sr(moi)
    assert r["kod"] == 3 and not r["diskretnoe"] and r["nepreryvnoe_pervoe"][0] == "tochka__S1__5"
    assert 0 < r["max_tochki"] < 1e-15
    assert _sr(moi, dopusk=1e-15)["kod"] == 0           # допуск владельца принимает 1 ulp, max |Δ| всё равно виден
    assert _sr(moi, dopusk=1e-15)["max_tochki"] > 0


def test_yakor_dopusk_1e12_fiksirovan():
    moi = _nabor()
    moi["yakor__S1"]["p"] = moi["yakor__S1"]["p"] + 1e-13
    assert _sr(moi)["kod"] == 0
    moi["yakor__S1"]["p"] = moi["yakor__S1"]["p"] + 1e-11
    assert _sr(moi)["kod"] == 3


def test_perevernutaya_sdelka_eto_diskretnoe_fail():
    moi = _nabor()
    moi["tochka__S1__5"]["sd"] = np.array([0, 1, 0], np.int64)
    assert _sr(moi, dopusk=1.0)["kod"] == 1             # даже огромный допуск не прощает дискретное


def test_sdvig_indeksa_vyhoda_i_pole_stroki_i_hashi_diskretnye():
    moi = _nabor()
    moi["tochka__S1__5"]["vi"] = np.array([[0, 1, 1], [0, 4, -1]], np.int64)
    assert _sr(moi)["kod"] == 1
    moi = _nabor()
    moi["tochka__S1__5"]["disk"] = '{"status": "ok", "n": 8, "sharpe": null}'
    assert _sr(moi)["kod"] == 1
    for k, v in (("kod", "k2"), ("kub", {"S1": "c2"}), ("daty", {"S1": "d2"})):
        assert _sr(_nabor(), meta={**META, k: v})["kod"] == 1


def test_nepolnyj_nabor_diskretnoe():
    moi = _nabor()
    del moi["tochka__S1__5"]
    assert _sr(moi)["kod"] == 1


def test_nan_protiv_chisla_ne_nol():
    assert es.delta(float("nan"), 1.0) == float("inf") and es.delta(float("nan"), float("nan")) == 0.0


def test_npz_tuda_obratno(tmp_path):
    p = str(tmp_path / "e.npz")
    ekviv_io.sohranit(p, _nabor(), META)
    rez, meta = ekviv_io.zagruzit(p)
    assert meta == META and es.sravnit(_nabor(), META, rez, meta)["kod"] == 0
    assert ekviv_io.tochki_iz(rez) == [("S1", 5)]
