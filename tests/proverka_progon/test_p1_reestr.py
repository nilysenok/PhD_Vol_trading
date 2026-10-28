"""П-1, лист Л-12: реестр отбора из ledger v2 — без пропусков, только дополнение. Красные до кода (Ж-081)."""
import csv
import gzip
import hashlib
import io

import pytest

from . import KVARTALY, KOD, VHODY, manifest, pisat, row, ryady


def _r():
    import fastbt.reestr_progon as m
    return m


def _ledger(tmp_path, lm, start=0):
    """4 ok + ne_schitana + oshibka + контроль + возмущение (shard=4 — закрывается каждые 4)."""
    pisat(lm, tmp_path, 0)
    w = lm.Pisatel(tmp_path, "S1", 0, manifest(), shard=4)
    kinds = [("ok", "sobol1", None)] * 3 + [("ne_schitana", "sobol1", None), ("oshibka", "sobol1", None),
                                              ("ok", "kontrol", "k1"), ("ok", "vozm", None), ("ok", "sobol2", None)]
    for n, (st, ist, ko) in enumerate(kinds):
        i = start + n
        r = row(lm, i, status=st, istochnik=ist, kontrol=ko)
        if ist == "vozm":
            r["vozm"] = {"komissiya": 0.0005}
        ok = st == "ok"
        w.add(r, *(ryady(i) if ok else (None, None)))
    w.close()
    return len(kinds)


def _tablica(path):
    raw = gzip.open(path, "rt", encoding="utf-8").read()
    return list(csv.DictReader(io.StringIO("".join(x for x in raw.splitlines(True) if not x.startswith("#")))))


@pytest.fixture
def vyhod(tmp_path, monkeypatch):
    m = _r()
    p = tmp_path / "reestr.csv.gz"
    monkeypatch.setattr(m, "VYHOD", str(p))
    return p


def test_kolonki_sostav():
    k = _r().kolonki()
    need = ["strategiya", "istochnik", "klyuch_sha", "klyuch", "kod", "kod_slit", "sloy", "status", "prichina",
            "mediana_kv", "dolya_pol_kv", "zaem5_mediana_kv", "zaem10_mediana_kv", "vyrozhdena", "kray",
            "region_id", "chlen_regiona", "kandidat", "wf_pravilo", "komissiya", "fdr"]
    need += [f"{p}_{y}" for p in ("sharpe", "sdelok", "msr") for y in range(2016, 2020)]
    need += [f"kv_{q}" for q in KVARTALY]
    assert set(need) <= set(k) and len(k) == len(set(k))
    assert sum(c.startswith("kv_") for c in k) == 16, "16 кварталов, не 15"


def test_vse_stroki_zapisany_vklyuchaya_ne_schitana_i_kontrol(tmp_path, vyhod):
    import fastbt.ledger_v2 as lm
    n = _ledger(tmp_path, lm)
    assert _r().dopisat_iz_ledger(tmp_path, "S1") == n
    t = _tablica(vyhod)
    assert len(t) == n
    assert {r["status"] for r in t} == {"ok", "ne_schitana", "oshibka"}
    assert {r["istochnik"] for r in t} >= {"sobol1", "sobol2", "kontrol", "vozm"}
    assert all(f"kv_{q}" in r for r in t for q in KVARTALY)
    assert list(t[0]) == _r().kolonki()


def test_povtor_nichego_ne_dobavlyaet(tmp_path, vyhod):
    import fastbt.ledger_v2 as lm
    n = _ledger(tmp_path, lm)
    m = _r()
    assert m.dopisat_iz_ledger(tmp_path, "S1") == n
    assert m.dopisat_iz_ledger(tmp_path, "S1") == 0
    assert len(_tablica(vyhod)) == n


def test_dopisyvanie_ne_perezapisyvaet_gzip_chitaetsya_celikom(tmp_path, vyhod):
    import fastbt.ledger_v2 as lm
    m = _r()
    n1 = _ledger(tmp_path, lm)
    m.dopisat_iz_ledger(tmp_path, "S1")
    pervye = vyhod.read_bytes()
    w = lm.Pisatel(tmp_path, "S1", 1, manifest(), shard=4)
    for i in range(100, 104):
        w.add(row(lm, i), *ryady(i))
    w.close()
    assert m.dopisat_iz_ledger(tmp_path, "S1") == 4
    posle = vyhod.read_bytes()
    assert posle.startswith(pervye) and len(posle) > len(pervye), "файл перезаписан, а не дополнен"
    assert len(_tablica(vyhod)) == n1 + 4


def test_otbor_dopisyvaetsya_novymi_strokami_status_otbor(tmp_path, vyhod):
    import fastbt.ledger_v2 as lm
    m = _r()
    n = _ledger(tmp_path, lm)
    m.dopisat_iz_ledger(tmp_path, "S1")
    sha = row(lm, 0)["klyuch_sha"]
    m.dopisat_iz_ledger(tmp_path, "S1", otbor={sha: {"region_id": "R1", "chlen_regiona": "1",
                                                      "kandidat": "1", "wf_pravilo": "A"}})
    t = _tablica(vyhod)
    assert len(t) == n + 1
    assert t[n]["status"] == "otbor" and t[n]["region_id"] == "R1" and t[n]["klyuch_sha"] == sha
    assert t[0]["status"] == "ok" and t[0]["region_id"] == ""


def test_ukazatel_v_reestr_A(tmp_path, vyhod, monkeypatch):
    import fastbt.ledger_v2 as lm
    import fastbt.reestr_otbora_v1 as v1
    m = _r()
    _ledger(tmp_path, lm)
    m.dopisat_iz_ledger(tmp_path, "S1")
    zapisi = []
    monkeypatch.setattr(v1, "dopisat", lambda rows, period, n, path=None: zapisi.append((rows, period, n)) or len(rows))
    m.ukazatel("S1")
    (rows, period, n), = zapisi
    assert period == "train4" and n == 1 and len(rows) == 1
    assert rows[0]["strategiya"] == "S1" and rows[0]["istochnik"] == "progon_v2"
    sha = hashlib.sha256(vyhod.read_bytes()).hexdigest()
    assert rows[0]["prichina"] == f"sha256 {sha}; строк {len(_tablica(vyhod))}"
