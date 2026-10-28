"""Тесты реестра прогона Л-12 (fastbt/reestr_progon.py) на синтетическом ledger v2 (только 2016–2019)."""
import csv
import gzip

import numpy as np
import pytest

from fastbt import ledger_v2 as L
from fastbt import reestr_otbora_v1 as V1
from fastbt import reestr_progon as R

T = 4
KOD, VHODY, SID = "K1", "V1", "S1"
MAN = {"T": T, "kod": KOD, "vhody": VHODY, "kod_slit": True}
KVS = [f"{g}Q{q}" for g in (2016, 2017, 2018, 2019) for q in range(1, 5)]


def _row(idx, status="ok", istochnik="sobol1", kv_keys=KVS):
    """Строка ledger §1.2: у ok — метрики и 16 кварталов, у ne_schitana — nan и пусто."""
    kl = L.klyuch(SID, {"i": idx}, "train4", 0.0004, None)
    ok = status == "ok"
    kv = {k: float(idx + n) for n, k in enumerate(kv_keys)} if ok else {}
    yrs = {str(y): float(idx) + 0.5 for y in (2016, 2017, 2018, 2019)} if ok else {}
    nan = float("nan")
    return {"v": "2.0", "format": "etap1-ledger", "strategiya": SID, "klyuch": kl,
            "klyuch_sha": L.klyuch_sha(kl, KOD, VHODY), "kod": KOD, "vhody": VHODY, "kod_slit": True,
            "period": "train4", "komissiya": 0.0004, "fdr": None, "status": status,
            "prichina": None if ok else "нет фильтра", "istochnik": istochnik, "sloy": "fast",
            "by_year": yrs, "sdelok_y": {k: 10.0 for k in yrs}, "msr_by_year": dict(yrs),
            "by_quarter": kv, "mediana_kv": 1.5 if ok else nan, "dolya_pol_kv": 0.5 if ok else nan,
            "by_quarter_zaem": {"5": kv, "10": kv} if ok else {},
            "flagi": {"vyrozhdena": False, "kray": ["k_s"], "short_shag1_kray": False}, "t_sek": 0.1}


def _zapis(root, specs, sid=SID, kv_keys=KVS):
    """specs: (idx, status, istochnik) — пишет через Pisatel и закрывает шард."""
    pi = L.Pisatel(root, sid, 1, MAN)
    for idx, status, istochnik in specs:
        pi.add(_row(idx, status, istochnik, kv_keys), np.arange(T, dtype=np.float64), np.zeros((17, T)))
    pi.close()


@pytest.fixture
def puti(tmp_path, monkeypatch):
    gz, a = tmp_path / "progon.csv.gz", tmp_path / "A_v1.csv"
    monkeypatch.setattr(R, "VYHOD", str(gz))
    monkeypatch.setitem(V1.VYHOD, "train4", str(a))
    return tmp_path / "led", gz, a


def _chitat(gz):
    with gzip.open(gz, "rt", encoding="utf-8") as f:
        return list(csv.DictReader(f))


SPEC5 = [(1, "ok", "sobol1"), (2, "ok", "sobol1"), (3, "ne_schitana", "sobol2"),
         (4, "ok", "kontrol"), (5, "oshibka", "sobol1")]


def test_vse_stroki_zapisany(puti):
    root, gz, _ = puti
    _zapis(root, SPEC5)
    assert R.dopisat_iz_ledger(root, SID) == 5
    rows = _chitat(gz)
    assert len(rows) == 5
    assert sorted(r["status"] for r in rows) == ["ne_schitana", "ok", "ok", "ok", "oshibka"]
    assert {r["istochnik"] for r in rows} == {"sobol1", "sobol2", "kontrol"}


def test_povtor_nichego_ne_dobavlyaet(puti):
    root, gz, _ = puti
    _zapis(root, SPEC5)
    R.dopisat_iz_ledger(root, SID)
    assert R.dopisat_iz_ledger(root, SID) == 0
    assert len(_chitat(gz)) == 5


def test_dva_dopisyvaniya_gzip_chitaetsya_tselikom(puti):
    root, gz, _ = puti
    _zapis(root, SPEC5)
    R.dopisat_iz_ledger(root, SID)
    before = gz.read_bytes()
    _zapis(root, [(6, "ok", "sobol2")])
    assert R.dopisat_iz_ledger(root, SID) == 1
    assert gz.read_bytes().startswith(before), "файл перезаписан, а не дописан"
    assert R.dopisat_iz_ledger(root, SID) == 0
    rows = _chitat(gz)
    assert len(rows) == 6
    assert rows[-1]["istochnik"] == "sobol2"
    with gzip.open(gz, "rt", encoding="utf-8") as f:
        assert f.read().count("strategiya,istochnik") == 1


def test_16_kvartalov_kolonki_i_znachenie(puti):
    root, gz, _ = puti
    _zapis(root, [(1, "ok", "sobol1")])
    R.dopisat_iz_ledger(root, SID)
    assert [c for c in R.kolonki() if c.startswith("kv_")] == [f"kv_{k}" for k in KVS]
    row = _chitat(gz)[0]
    assert float(row["kv_2016Q1"]) == 1.0 and float(row["kv_2019Q4"]) == 16.0
    assert row["mediana_kv"] == "1.5" and row["zaem5_mediana_kv"] != ""


def test_15_kvartalov_otkaz(puti):
    root, gz, _ = puti
    _zapis(root, [(1, "ok", "sobol1")], kv_keys=KVS[:15])
    with pytest.raises(ValueError, match="16"):
        R.dopisat_iz_ledger(root, SID)
    assert not gz.exists(), "при отказе файл не должен появиться"


def test_otbor_novymi_strokami_bez_izmeneniya_starych(puti):
    root, gz, _ = puti
    _zapis(root, SPEC5)
    R.dopisat_iz_ledger(root, SID)
    stare = _chitat(gz)
    sha = _row(1)["klyuch_sha"]
    meta = {sha: {"region_id": "R1", "chlen_regiona": "1", "kandidat": "1", "wf_pravilo": "w1"}}
    R.dopisat_iz_ledger(root, SID, otbor=meta)
    R.dopisat_iz_ledger(root, SID, otbor=meta)
    rows = _chitat(gz)
    assert rows[:5] == stare
    otb = [r for r in rows if r["status"] == "otbor"]
    assert len(otb) == 1 and otb[0]["region_id"] == "R1" and otb[0]["klyuch_sha"] == sha


def test_otbor_neizvestnyj_klyuch_otkaz(puti):
    root, _, _ = puti
    _zapis(root, [(1, "ok", "sobol1")])
    with pytest.raises(ValueError, match="неизвестного"):
        R.dopisat_iz_ledger(root, SID, otbor={"0" * 64: {"region_id": "R1"}})


def test_ukazatel_pishet_stroku_v_A_v1(puti):
    root, gz, a = puti
    _zapis(root, SPEC5)
    R.dopisat_iz_ledger(root, SID)
    row = R.ukazatel(SID)
    assert row["istochnik"] == "progon_v2" and row["strategiya"] == SID
    assert row["prichina"].startswith("sha256 ") and row["prichina"].endswith("строк 5")
    R.ukazatel(SID)
    with open(a, encoding="utf-8") as f:
        data = [ln for ln in f if not ln.startswith("#")]
    assert len(list(csv.DictReader(data))) == 1
