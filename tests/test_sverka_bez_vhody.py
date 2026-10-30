"""Сверка Мак ↔ сервер: vhody (хэш float-массивов, SVML x86 ≠ libm arm64) не сверяется; сделки, sha кода и данных — строго."""
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/server"))
import ekviv  # noqa: E402
import ekviv_sravn as es  # noqa: E402

META = dict(kod="k1", kub={"S1": "c1"}, daty={"S1": "d1"}, vhody={"S1": "v1"}, v6_vhody={"S1": "w1"},
            kod_sha="tar1", dannye_sha="man1")


def _zapis(vhody="v1", sdelok=2):
    row = {"status": "ok", "n": sdelok, "sharpe": 1.5, "vhody": vhody}
    return dict(sig=json.dumps(row), disk=ekviv.json.dumps(ekviv._bez_vesh(_bez(row)), sort_keys=True),
                hex={"/sharpe": (1.5).hex()}, p=np.array([0.0, 0.01]), t=np.array([0.1]), pz=np.array([0.0, 1.0]),
                sd=np.array([0, sdelok, 0], np.int64), vi=np.array([[0, 1, 1], [0, 3, -1]], np.int64))


def _bez(row):      # как ekviv._zapis: vhody в disk не попадает
    return {k: v for k, v in row.items() if k != "vhody"}


def _sr(moi, meta_moi, eta=None):
    return es.sravnit(eta or {"yakor__S1": _zapis()}, META, moi, meta_moi, 1e-12)


def test_otlichaetsya_tolko_vhody_ok():
    m = copy.deepcopy(META)
    m["vhody"], m["v6_vhody"] = {"S1": "x86"}, {"S1": "x86"}
    r = _sr({"yakor__S1": _zapis(vhody="x86")}, m)
    assert r["kod"] == 0 and not r["diskretnoe"]


def test_zapis_ekviv_ne_beryot_vhody_v_disk(monkeypatch):
    class PS:
        SVERKA_PROPUSK = {"t_sek", "kod", "kod_slit", "klyuch_sha", "excess", "sloy"}
    monkeypatch.setattr(ekviv, "_ps", lambda: PS)
    z = ekviv._zapis({"status": "ok", "vhody": "abc", "n": 3}, None, None, None)
    assert "vhody" not in json.loads(z["disk"])


def test_otlichaetsya_chislo_sdelok_fail():
    m = copy.deepcopy(META)
    moi = {"yakor__S1": _zapis(sdelok=3)}
    moi["yakor__S1"]["sd"] = np.array([0, 3, 0], np.int64)
    assert _sr(moi, m)["kod"] == 1


def test_otlichaetsya_sha_dannyh_ili_koda_fail():
    for k in ("dannye_sha", "kod_sha", "kod"):
        m = copy.deepcopy(META)
        m[k] = "drugoj"
        r = _sr({"yakor__S1": _zapis()}, m)
        assert r["kod"] == 1 and k in r["diskretnoe"][0], k


def test_mutaciya_vhody_snova_sveryaetsya_lovitsya(monkeypatch):
    """Если SHAPKA снова содержит vhody — тест «только vhody» падает: мутация поймана."""
    monkeypatch.setattr(es, "SHAPKA", es.SHAPKA + ("vhody", "v6_vhody"))
    m = copy.deepcopy(META)
    m["vhody"] = {"S1": "x86"}
    assert _sr({"yakor__S1": _zapis(vhody="x86")}, m)["kod"] == 1
