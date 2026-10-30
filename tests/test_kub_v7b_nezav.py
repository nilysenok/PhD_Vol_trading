"""Независимый тестировщик #914 (кубы v7 S8–S12, canon_v7b; Ж-153 пп. 1–4, Ж-156 п. 3, Ж-157). Данных не читает, сети нет.
Ожидания заданы явно (из постановки), не вычислены кодом canon_v7b. Мутации — проверки-функции на подложенных копиях."""
from __future__ import annotations

import copy
import itertools
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from fastbt import canon_v6 as c6, canon_v7 as c7, canon_v7b as b

ROOT = Path(c7.ROOT)
V7 = ROOT / "fastbt/designs/v7"
IMP, SECH = ("S10", "S11", "S12"), ("S8", "S9")
OSI_S3 = ["stop_nach", "tejk", "trail", "chast", "razmer_tip", "m_sily", "rang_tip", "sigma_ist", "bezubytok", "stop_v_plyus", "pir"]


@pytest.fixture(scope="module")
def kuby():
    return {s: b.load_kub(s) for s in SECH + IMP}


def _canony(sid, k, n, seed):
    ox = b.osi(k)
    rng = np.random.default_rng(seed)
    return [b.canon_v7b(sid, c7.tochka(ox, rng.random(len(ox))), k) for _ in range(n)]


# ---------- проверки (их же роняют мутации) ----------
def proverka_reshetka(kuby, canon):
    for sid in SECH:
        k = kuby[sid]
        assert [list(v["setka"]) for v in k["signal"].values()] == [[0.3, 0.5, 0.7], [0, 0.1, 0.2], [2, 3, 4], ["den", "nedelya"]], sid
        assert list(k["signal"]) == ["porog", "bufer", "cap", "rebal"] and k["osi7"]["k_okup"]["setka"] == ["net", 3]
        assert b.obem(k) == 108, sid
        grid = {n: v["setka"] for n, v in k["signal"].items()} | {"k_okup": ["net", 3]}
        tochki = {tuple(sorted((a, str(v)) for a, v in canon(sid, dict(zip(grid, p)), k).items()))
                  for p in itertools.product(*grid.values())}
        assert len(tochki) == 108, (sid, len(tochki))                    # все различны, ни одна не выброшена


def proverka_n1_imp(kuby):
    for sid in IMP:
        assert kuby[sid]["sobol"]["N"] == 200_000, sid


def proverka_tau_vhod_T(kuby, canon):
    for sid in IMP:
        k = kuby[sid]
        s = k["signal"]
        assert s["tau"]["setka"] == [2, 3, 5] and 1 not in s["tau"]["setka"] and s["tau"].get("spravka") == 1, sid
        assert s["vhod"]["setka"] == ["d1", "T"] and s["T"]["setka"] == ["17:00", "18:00", "18:40"], sid
        assert s["T"]["roditel"] == {"vhod": "T"}, sid
        ox = b.osi(k)
        rng, kv, kt = np.random.default_rng(14), set(), set()
        for _ in range(3000):
            c = canon(sid, c7.tochka(ox, rng.random(len(ox))), k)
            if c is None:
                continue
            assert c["tau"] in (2, 3, 5) and c["vhod"] in ("d1", "T"), sid
            if c["vhod"] == "d1":
                assert "T" not in c, (sid, c)
            else:
                assert c.get("T") in ("17:00", "18:00", "18:40"), (sid, c)
            kv.add(c["vhod"])
            kt.add(c.get("T"))
        assert kv == {"d1", "T"} and kt == {None, "17:00", "18:00", "18:40"}, sid


# ---------- S8 / S9 ----------
def test_s8_s9_reshetka_108(kuby):
    proverka_reshetka(kuby, b.canon_v7b)


def test_s8_s9_net_pravil_nesovmestimosti(kuby):
    for sid in SECH:
        assert all(c is not None for c in _canony(sid, kuby[sid], 500, 8)), sid


def test_s8_s9_ravnomernyj_vybor_pokryvaet_reshetku(kuby):
    for sid in SECH:
        cs = {tuple(sorted((a, str(v)) for a, v in c.items())) for c in _canony(sid, kuby[sid], 6000, 9)}
        assert len(cs) == 108, sid


def test_s8_s9_net_osi_otnositelno_i_k_okup_dva(kuby):
    for sid in SECH:
        assert set(kuby[sid]["osi7"]) == {"k_okup"} and kuby[sid]["osi7"]["k_okup"]["setka"] == ["net", 3], sid


# ---------- S10–S12 ----------
def test_s10_s12_tau_vhod_T(kuby):
    proverka_tau_vhod_T(kuby, b.canon_v7b)


def test_s10_s12_n1_200000(kuby):
    proverka_n1_imp(kuby)


def proverka_k_okup_net_ubran(kuby, canon):
    for sid in IMP:
        k = kuby[sid]
        ox = b.osi(k)
        rng = np.random.default_rng(10)
        cs = [canon(sid, c7.tochka(ox, rng.random(len(ox))), k) for _ in range(800)]
        cs = [c for c in cs if c is not None]
        assert {c.get("k_okup", "net") for c in cs} == {"net", 3} and all(c.get("k_okup") != "net" for c in cs), sid


def test_s10_s12_k_okup_net_i_tri_i_net_ubiraetsya(kuby):
    assert all(kuby[s]["osi7"]["k_okup"]["setka"] == ["net", 3] for s in IMP)
    proverka_k_okup_net_ubran(kuby, b.canon_v7b)


def test_s10_s12_obshchie_osi_ravny_s3_v6(kuby):
    s3 = {n: g for n, g, _ in c6.osi(c6.load_kub("S3"), None)}
    for sid in IMP:
        k = kuby[sid]
        assert k["osi_s3"] == OSI_S3 and k["baza"]["iz"] == "S3" and k["baza"]["kub_sha"] == c7.sha_v6("S3"), sid
        got = {n: g for n, g, _ in b.osi(k, None)}
        for n in OSI_S3:
            assert got[n] == s3[n], (sid, n)
        assert set(got) == set(OSI_S3) | {"tau", "vhod", "T", "k_okup"}, sid


def test_s10_s12_ne_vkhodyat_osi_vne_perechnya(kuby):
    zapret = {"k_e", "t_vyhod", "time_decay", "rel_signal", "hurst_th", "reentry", "vol_regime_max"}
    for sid in IMP:
        for c in _canony(sid, kuby[sid], 300, 12):
            assert c is None or not (zapret & set(c)), (sid, zapret & set(c))


def test_s10_s12_pravila_trenda_v_kanone(kuby):
    """Точки, несовместимые по canon_v6 (S3), выбрасываются и здесь."""
    for sid in IMP:
        k = kuby[sid]
        cs = _canony(sid, k, 2000, 15)
        for c in cs:
            if c is not None:
                rest = {n: v for n, v in c.items() if n not in ("tau", "vhod", "T", "k_okup")}
                assert c6.canon_v6("S3", rest, k["s3"]) is not None, (sid, c)


def test_proverit_otkaz_na_chuzhom_sha_i_k_okup(kuby):
    for sid in IMP:
        m = copy.deepcopy(kuby[sid])
        m["baza"]["kub_sha"] = "0" * 64
        with pytest.raises(ValueError):
            b.proverit(m)
    m = copy.deepcopy(kuby["S8"])
    m["osi7"]["k_okup"]["setka"] = ["net", 2, 3]
    with pytest.raises(ValueError):
        b.proverit(m)


def test_sha256sums_vse_odinnadcat_faylov():
    assert c7.sha_nesovpadenia() == []
    assert {f.name for f in V7.glob("kub_S*.yaml")} == {f"kub_S{i}.yaml" for i in (1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12)}


def test_seed_v7_novye_ne_ravny_v6_dlya_s10_s12(kuby):
    s6 = yaml.safe_load((ROOT / "fastbt/designs/v6/kub_S3.yaml").read_text(encoding="utf-8"))["sobol"]
    for sid in SECH + IMP:
        s7 = kuby[sid]["sobol"]
        assert {s7["seed"], s7["seed2"], s7["morris_seed"]}.isdisjoint({s6["seed"], s6["seed2"], s6["morris_seed"]}), sid


def test_s7_net_v_v7():
    assert not (V7 / "kub_S7.yaml").exists() and "S7" not in b.SIDS_B


_SKRIPT = ("import hashlib,json,numpy as np;from fastbt import canon_v7 as c7, canon_v7b as b;h=hashlib.sha256();\n"
           "for s in ('S8','S10','S12'):\n k=b.load_kub(s);ox=b.osi(k);r=np.random.default_rng(3)\n"
           " for _ in range(500):h.update(json.dumps(b.canon_v7b(s,c7.tochka(ox,r.random(len(ox))),k),sort_keys=True).encode())\n"
           "print(h.hexdigest())")


def test_determinizm_dva_processa_raznyj_hashseed():
    res = []
    for hs in ("2", "424242"):
        r = subprocess.run([sys.executable, "-c", _SKRIPT], cwd=ROOT, env=dict(os.environ, PYTHONHASHSEED=hs),
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-500:]
        res.append(r.stdout.strip())
    assert res[0] == res[1] and len(res[0]) == 64


# ---------- мутации ----------
def test_mut_T_ne_svorachivaetsya_pri_d1_lovitsya(kuby):
    def bez_svertki(sid, pt, k):
        c = b.canon_v7b(sid, pt, k)
        if c is not None and c.get("vhod") == "d1":
            c = dict(c, T=pt["T"])
        return c
    with pytest.raises(AssertionError):
        proverka_tau_vhod_T(kuby, bez_svertki)


def test_mut_T_propadaet_pri_T_lovitsya(kuby):
    def teryaet(sid, pt, k):
        c = b.canon_v7b(sid, pt, k)
        return None if c is None else {a: v for a, v in c.items() if a != "T"}
    with pytest.raises(AssertionError):
        proverka_tau_vhod_T(kuby, teryaet)


def test_mut_tau_1_v_setke_lovitsya(kuby):
    m = copy.deepcopy(kuby)
    m["S11"]["signal"]["tau"]["setka"] = [1, 2, 3, 5]
    with pytest.raises(AssertionError):
        proverka_tau_vhod_T(m, b.canon_v7b)


def test_mut_s10_n1_100000_lovitsya(kuby):
    m = copy.deepcopy(kuby)
    m["S10"]["sobol"]["N"] = 100_000
    with pytest.raises(AssertionError):
        proverka_n1_imp(m)


def test_mut_reshetka_bez_k_okup_lovitsya(kuby):
    m = copy.deepcopy(kuby)
    m["S9"]["osi7"]["k_okup"]["setka"] = ["net", 2, 3]
    with pytest.raises(AssertionError):
        proverka_reshetka(m, b.canon_v7b)


def test_mut_reshetka_vybrasyvaet_tochku_lovitsya(kuby):
    def vybros(sid, pt, k):
        return None if pt["cap"] == 4 and pt["rebal"] == "den" else b.canon_v7b(sid, pt, k)
    with pytest.raises((AssertionError, AttributeError)):
        proverka_reshetka(kuby, vybros)


def test_mut_k_okup_net_ostayotsya_v_kanone_lovitsya(kuby):
    def derzhit(sid, pt, k):
        c = b.canon_v7b(sid, pt, k)
        return None if c is None else dict(c, k_okup=pt["k_okup"])
    with pytest.raises(AssertionError):
        proverka_k_okup_net_ubran(kuby, derzhit)
