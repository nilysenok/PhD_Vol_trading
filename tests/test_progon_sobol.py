"""Л-6: драйвер Соболя на синтетическом оценщике (данных нет). Мутации — в test_mut_*."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import fastbt.ledger_v2 as lm
import fastbt.progon_sobol as ps

T = 8


class Fake:
    """Подставной evaluate_v2: ряды зависят только от точки; tik во float32-представимых значениях."""
    def evaluate_v2(self, pt, period):
        assert period == "train4"
        rng = np.random.default_rng(int(lm.klyuch_sha(lm.klyuch("S1", pt, period, 0.0004, None), "a", "b")[:8], 16))
        tik = rng.normal(0, 0.01, (17, T)).astype(np.float32).astype(np.float64)
        by = {str(y): 0.5 for y in range(2016, 2020)}
        return {"by_year": by, "T": T, "mediana_kv": 0.1}, np.nanmean(tik, axis=0), tik


def sozdat(ostatok=Fake):
    daty = np.arange(np.datetime64("2016-01-04"), np.datetime64("2016-01-04") + T)
    return lambda sid: ps.Sreda(ostatok(), "kod-fake", "vhody-fake", daty, [f"T{i:02d}" for i in range(17)])


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(ps, "fabrika", sozdat())
    monkeypatch.setattr(ps, "_W", {})


def stroki(root, sid):
    out = []
    for p in sorted((Path(root) / sid).glob("p[0-9]*/rows.jsonl")):
        out += [json.loads(x) for x in p.read_text().splitlines()]
    return out


def test_prohod2_prodolzhenie_i_zakonchennost_kuba():
    c = ps.load_kub("S1")
    assert ps.tochki(c, 0, 5000)[4990:] == ps.tochki(c, 4990, 5000)
    assert ps.tochki(c, 5000, 5010)[0][0] == 5000
    assert ps.tochki(c, 5000, 5010) != ps.tochki(c, 0, 10)       # мутация «проход 2 с 0»


def test_kanon_primenyon_k_s_net_i_k_x():
    for sid in ("S1", "S4"):
        for _, p in ps.tochki(ps.load_kub(sid), 0, 1000):
            if p["k_s"] == "net":
                assert p["R"] == p["k_tr"] == p["bezubytok"] == "net"
            else:
                assert p["k_x"] == "net"


def test_chislo_unikalnyh_ne_zavysheno():
    """N = различные klyuch_sha: canon_v2 схлопывает R/k_tr/bezubytok при k_s = net."""
    c = ps.load_kub("S1")
    pts = [p for _, p in ps.tochki(c, 0, 1000)]
    raw = [lm.klyuch("S1", p, "train4", 0.0004, None) for p in pts]
    assert len({json.dumps(k, sort_keys=True) for k in raw}) == len(pts)
    assert all(lm.canon_v2("S1", p) == p for p in pts)


def test_schet_rezyum_bez_dublej_i_poter(fake, tmp_path):
    a = ps.schet(["S1", "S2"], 0, 150, "sobol1", tmp_path, 2, False, False)
    assert a["S1"]["n"] == a["S2"]["n"] == 150 and a["S1"]["oshibok"] == 0
    b = ps.schet(["S1", "S2"], 0, 250, "sobol1", tmp_path, 2, False, False)
    assert b["S1"]["n"] == b["S2"]["n"] == 100                    # 150 уже закрыты — не пересчитаны
    for sid in ("S1", "S2"):
        rows = stroki(tmp_path, sid)
        shas = [r["klyuch_sha"] for r in rows]
        assert len(shas) == len(set(shas)) == 250
        assert sorted(r["sobol_i"] for r in rows) == list(range(250))
        assert lm.Chitatel(tmp_path, sid).n_unique() == 250
    c = ps.schet(["S1"], 0, 250, "sobol1", tmp_path, 1, False, False)
    assert c["S1"]["n"] == 0 and len(stroki(tmp_path, "S1")) == 250


def test_kod_slit_po_umolchaniyu_false_i_flag(fake, tmp_path):
    ps.schet(["S1"], 0, 20, "sobol1", tmp_path / "a", 1, False, False)
    ps.schet(["S1"], 0, 20, "sobol2", tmp_path / "b", 1, True, False)
    assert {r["kod_slit"] for r in stroki(tmp_path / "a", "S1")} == {False}
    assert {r["kod_slit"] for r in stroki(tmp_path / "b", "S1")} == {True}
    assert {r["istochnik"] for r in stroki(tmp_path / "b", "S1")} == {"sobol2"}


def test_stroka_kontrakt_i_hash_koda(fake, tmp_path):
    ps.schet(["S1"], 0, 10, "sobol1", tmp_path, 1, False, False)
    r = stroki(tmp_path, "S1")[0]
    assert r["kod"] == "kod-fake" and r["sloy"] == "slow" and r["period"] == "train4" and r["komissiya"] == 0.0004
    assert r["klyuch"]["komissiya"] == "0.0004" and r["klyuch"]["period"] == "train4" and r["fdr"] is None
    assert (tmp_path / "S1" / "daty.npy").exists()


def test_data_2020_otkaz(monkeypatch, tmp_path):
    daty = np.arange(np.datetime64("2019-12-30"), np.datetime64("2019-12-30") + T)
    monkeypatch.setattr(ps, "fabrika", lambda sid: ps.Sreda(Fake(), "k", "v", daty, []))
    monkeypatch.setattr(ps, "_W", {})
    with pytest.raises(ValueError, match="2020"):
        ps.schet(["S1"], 0, 5, "sobol1", tmp_path, 1, False, False)


def test_period_i_komissiya_otkaz():
    c = ps.load_kub("S1")
    for k, v in (("period", "train"), ("komissiya", 0.0005), ("fdr", "t0")):
        with pytest.raises(ValueError):
            ps.tochki({**c, k: v}, 0, 2)


def test_processov_bolshe_10_otkaz(tmp_path):
    with pytest.raises(ValueError):
        ps.schet(["S1"], 0, 5, "sobol1", tmp_path, 11, False, False)


def test_disk_otkaz_pri_nehvatke(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(10**12, 10**12 - 10**9, 10**9))
    with pytest.raises(SystemExit):
        ps.proverit_disk(1, tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(10**12, 0, 10**12))
    ps.proverit_disk(1, tmp_path)
    with pytest.raises(SystemExit):                              # 1,5 × оценки больше свободного
        ps.proverit_disk(10**7, tmp_path)


def test_sverka_zelenaya_i_lovit_raskhozhdenie(fake, monkeypatch, tmp_path):
    ps.schet(["S1", "S2"], 0, 30, "sobol1", tmp_path, 2, False, False)
    monkeypatch.setattr(ps, "SIDS", ["S1", "S2"])
    assert ps.sverka(20, tmp_path, False) == 0

    class Sdvig(Fake):
        def evaluate_v2(self, pt, period):
            r, p, t = super().evaluate_v2(pt, period)
            return r, p + 1e-9, t
    monkeypatch.setattr(ps, "fabrika", sozdat(Sdvig))
    monkeypatch.setattr(ps, "_W", {})
    assert ps.sverka(20, tmp_path, False) == 1


def test_seed_po_strategiyam():
    assert [ps.load_kub(f"S{k}")["sobol"]["seed"] for k in range(1, 7)] == [20261028 + k for k in range(1, 7)]
