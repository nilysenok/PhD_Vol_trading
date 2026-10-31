"""Пилот-1 в progon_v7 (Ж-160 п. 3): --semeystva, --bez-osej (ось фиксируется в «выкл./net/0» до оценки), пометки в pilot_zamer.json и manifest,
p50/p90 секунд на точку. Синтетика, без сети и данных; точки, дошедшие до оценщика, пишет сам оценщик (дети fork — через файл)."""
import json
import shutil

import numpy as np
import pytest

import fastbt.period_v2 as pv
import fastbt.progon_sobol as ps
import fastbt.progon_v7 as p7
from test_progon_server import Fake, syrye

SID, T, N = "S1", 8, 40
DATY = np.arange(np.datetime64("2017-01-09"), np.datetime64("2017-01-09") + T)


class Pishet(Fake):
    fdr = None
    zhurnal = None

    def evaluate_v2(self, pt, period):
        with open(self.zhurnal, "a", encoding="utf-8") as f:
            f.write(json.dumps(pt, sort_keys=True) + "\n")
        return super().evaluate_v2(pt, period)


def tochki_fn(kub, n0, n1, rz):
    return [(i, {"a": float(i), "min_uderzhanie": 2, "k_okup": 3}) for i in range(n0, n1)]


@pytest.fixture
def sred(monkeypatch, tmp_path):
    (tmp_path / "kub_S1.yaml").write_text("sid: S1\n", encoding="utf-8")
    monkeypatch.setattr(p7, "KUB7", tmp_path / "kub_{sid}.yaml")
    monkeypatch.setattr(ps, "_W", {})
    monkeypatch.setattr(pv, "proverit_chtenie", lambda *a, **k: None)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(10**13, 0, 10**13))
    Pishet.zhurnal = tmp_path / "ocenennye.jsonl"
    monkeypatch.setattr(ps, "fabrika", lambda sid: ps.Sreda(Pishet(), "kod-fake", "vhody-fake", DATY, [f"T{i:02d}" for i in range(17)]))
    return tmp_path


def kubs(tmp_path):
    c = {"sid": SID, "setka": "v7", "fdr": None, "data": "2026-10-31", "sobol": {"N": N, "seed": 1, "seed2": 2}}
    return {SID: (tmp_path / "kub_S1.yaml", c)}


def pilot(tmp_path, **kw):
    out = tmp_path / "out"
    res = p7.pilot(kubs(tmp_path), out, 2, {SID: N}, tochki_fn=tochki_fn, rezhim="prezhniy", shag=100, **kw)
    ocen = [json.loads(x) for x in Pishet.zhurnal.read_text(encoding="utf-8").splitlines()]
    return out, res, ocen


def manifest(out):
    return json.loads((out / SID / "manifest.json").read_text(encoding="utf-8"))


def test_g1_bez_osej_min_uderzhanie_net_konstant(sred):
    out, res, ocen = pilot(sred, bez_osej=("min_uderzhanie",), semeystva=(SID,))
    assert len(ocen) == N and all(p["min_uderzhanie"] == 0 for p in ocen)          # константа ≠ 0 не дошла до оценщика
    assert all(p["k_okup"] == 3 and "a" in p for p in ocen)                         # чужие оси не тронуты
    z = json.loads((out / p7.ZAMER).read_text(encoding="utf-8"))[SID]
    assert (z["pilot"], z["osi_otsutstvuyut"], z["semeystva"]) == ("пилот-1", ["min_uderzhanie"], [SID])
    mf = manifest(out)
    assert (mf["pilot"], mf["osi_otsutstvuyut"], mf["semeystva"]) == ("пилот-1", ["min_uderzhanie"], [SID])


def test_k_okup_net_stroka_i_stroka_ledger_po_fakticheskoj_tochke(sred):
    out, _, ocen = pilot(sred, bez_osej=("k_okup", "min_uderzhanie"))
    assert {p["k_okup"] for p in ocen} == {"net"} and {p["min_uderzhanie"] for p in ocen} == {0}
    assert '"net"' in json.dumps(syrye(out, SID)[0], ensure_ascii=False) or "net" in json.dumps(syrye(out, SID)[0], ensure_ascii=False)


def test_g2_mutatsiya_os_ne_fiksiruetsya_lovitsya(sred, monkeypatch):
    monkeypatch.setattr(p7, "fiksirovat", lambda tf, osi: tf)                       # подложено: ось не фиксируется
    _, _, ocen = pilot(sred, bez_osej=("min_uderzhanie",))
    assert not all(p["min_uderzhanie"] == 0 for p in ocen)                           # проверка G1 на мутации падает


def test_bez_osej_pusto_tochki_kak_v_v6(sred):
    out, _, ocen = pilot(sred)
    assert all(p["min_uderzhanie"] == 2 and p["k_okup"] == 3 for p in ocen)
    assert manifest(out)["osi_otsutstvuyut"] == [] and manifest(out)["pilot"] == "пилот-1"


def test_g3_semeystva_tolko_s1():
    baza = ["--kub", "v7", "--period", "polnaya", "--out", "x"]
    a = p7.razbor(baza + ["--semeystva", "S1"])
    assert a.strategii == "S1" and p7.sids_iz(a) == ["S1"]
    assert p7.razbor(baza + ["--bez-osej", "min_uderzhanie,k_okup"]).bez_osej == "min_uderzhanie,k_okup"
    with pytest.raises(SystemExit, match="S7"):
        p7.sids_iz(p7.razbor(baza + ["--semeystva", "S1,S7"]))


def test_g4_zamer_p50_p90_rss_razmer(sred):
    _, res, _ = pilot(sred, bez_osej=("min_uderzhanie",), semeystva=(SID,))
    z = res[SID]
    assert z["n"] == N and 0 < z["sek_na_tochku_p50"] <= z["sek_na_tochku_p90"]
    assert z["rss_pik_mb"] > 0 and z["ledger_bajt_na_1000"] == int(z["ledger_bajt"] * 1000 / N)
    assert z["ledger_gb_na_1000"] == z["ledger_bajt_na_1000"] / 1e9


def test_kvantili_chistaya_funkciya():
    assert p7.kvantili([1.0, 2.0, 3.0, 4.0, 5.0]) == (3.0, pytest.approx(4.6))
    assert p7.kvantili([]) == (None, None)
