"""Независимый тестировщик #909 (кубы v7 S1–S6, canon_v7; Ж-153 п. 1.1, Ж-155 п. 2, Ж-156 п. 3, Ж-157 п. 2). Данных не читает, сети нет.
Ожидания считаются заново из yaml куба v6 (центр = setka[len // 2], соседи по порядку yaml) и не берутся из кода canon_v7.
Мутации — проверки-функции, поданные на подложенные копии (в памяти / во временном каталоге), не на ветку."""
from __future__ import annotations

import copy
import functools
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from fastbt import canon_v6 as c6, canon_v7 as c7

SIDS = [f"S{i}" for i in range(1, 7)]
ROOT = Path(c7.ROOT)
V6, V7 = ROOT / "fastbt/designs/v6", ROOT / "fastbt/designs/v7"
N1 = {"S1": 100_000, "S2": 100_000, "S3": 100_000, "S4": 100_000, "S5": 100_000, "S6": 50_000}
OSI7 = {"S1": ("k_okup", "otnositelno"), "S2": ("k_okup", "otnositelno"), "S3": ("k_okup", "otnositelno"),
        "S4": ("k_okup", "otnositelno"), "S5": ("k_okup",), "S6": ("k_okup",)}


def _yml(p):
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=None)
def _v6_grids_c(sid):
    return {n: list(g) for n, g, _ in c6.osi(c6.load_kub(sid), None)}


def _v6_grids(sid):
    """Сетки всех осей куба v6 по имени (номинал, без фильтра готовности); копия из кэша."""
    return copy.deepcopy(_v6_grids_c(sid))


def _ozhid(g):
    i = len(g) // 2
    return g[max(0, i - 1): i + 2]


@pytest.fixture(scope="module")
def kuby():
    return {s: c7.load_kub(s) for s in SIDS}


# ---------- проверки (функции от набора кубов — их же роняют мутации) ----------
def proverka_signal(kuby):
    for sid, k in kuby.items():
        base = _v6_grids(sid)
        assert list(k["signal"]) == list(c7.c5.SIGNAL[sid]), sid
        for n, v in k["signal"].items():
            assert v["setka"] == _ozhid(base[n]), (sid, n, v["setka"], base[n])
            assert v["centr"] == base[n][len(base[n]) // 2], (sid, n)
            if len(base[n]) == 2:
                assert v["setka"] == base[n], (sid, n, "ось из 2 значений — оба")
            else:
                assert len(v["setka"]) == 3 and v["setka"][1] == v["centr"], (sid, n)


def proverka_k_okup(kuby):
    for sid, k in kuby.items():
        ko = k["osi7"]["k_okup"]
        assert ko["setka"] == ["net", 3] and ko["vykl"] == "net", sid
        assert set(k["osi7"]) == set(OSI7[sid]), (sid, list(k["osi7"]))


def proverka_n1(kuby):
    for sid, k in kuby.items():
        assert k["sobol"]["N"] == N1[sid], (sid, k["sobol"]["N"])


def proverka_dvojnoj_fon(kuby, canon):
    for sid in ("S1", "S2", "S3", "S4"):
        k = kuby[sid]
        ox = c7.osi(k, c7.GOTOVY_V7)
        rng = np.random.default_rng(31)
        for _ in range(200):
            pt = c7.tochka(ox, rng.random(len(ox)))
            for rel, otn in itertools.product(("syroy", "sektor", "rynok"), ("net", "sektor", "imoex")):
                if otn != "net" and rel != "syroy":
                    assert canon(sid, dict(pt, rel_signal=rel, otnositelno=otn), k) is None, (sid, rel, otn)
        ok = [canon(sid, dict(c7.tochka(ox, rng.random(len(ox))), rel_signal="syroy", otnositelno="sektor"), k) for _ in range(200)]
        assert any(c is not None and c.get("otnositelno") == "sektor" for c in ok), sid


# ---------- Ж-153 п. 1.1 ----------
def test_signal_centr_plyus_minus_odin_po_poryadku_yaml(kuby):
    proverka_signal(kuby)


def test_signal_yavnye_znacheniya_s1_s6(kuby):
    s1 = kuby["S1"]["signal"]
    assert s1["ma_tip"]["setka"] == ["wma", "hma", "dema"] and s1["ma_window"]["setka"] == [25, 30, 35]
    assert s1["z_entry"]["setka"] == [1.5, 1.75, 2.0] and s1["dev_tip"]["setka"] == ["std", "atr"]
    s6 = kuby["S6"]["signal"]
    assert s6["dir_bias"]["setka"] == ["OFF", "ON"] and s6["vwap_window"]["setka"] == [18, 20, 25]
    assert s6["filter_set"]["setka"] == [1, 2, 3]


# ---------- Ж-156 п. 3 ----------
def test_k_okup_net_i_tri(kuby):
    proverka_k_okup(kuby)


def test_v6_vnutri_v7_klyuch_tochki_v6(kuby):
    """Точка v7 с выкл. осями v7 и сигналом в пределах v6: канон v7 == канон v6, ключей v7 нет, json-хэш тот же."""
    for sid, k in kuby.items():
        ox = c7.osi(k, c7.GOTOVY_V7)
        rng, n_ok = np.random.default_rng(909), 0
        for _ in range(400):
            pt = c7.tochka(ox, rng.random(len(ox)))
            pt["k_okup"] = "net"
            if "otnositelno" in pt:
                pt["otnositelno"] = "net"
            c = c7.canon_v7(sid, pt, k)
            ref = c6.canon_v6(sid, {a: b for a, b in pt.items() if a not in ("k_okup", "otnositelno")}, k["v6"])
            assert c == ref, sid
            if c is None:
                continue
            n_ok += 1
            assert "k_okup" not in c and "otnositelno" not in c, sid
            assert json.dumps(c, sort_keys=True) == json.dumps(ref, sort_keys=True)
            for n in k["signal"]:
                assert c[n] in _v6_grids(sid)[n], (sid, n)
        assert n_ok > 50, sid


def test_k_okup_tri_ne_ubiraetsya(kuby):
    k = kuby["S5"]
    ox = c7.osi(k, c7.GOTOVY_V7)
    rng = np.random.default_rng(5)
    kv = [c7.canon_v7("S5", dict(c7.tochka(ox, rng.random(len(ox))), k_okup=3), k) for _ in range(300)]
    kv = [c for c in kv if c is not None]
    assert kv and all(c["k_okup"] == 3 for c in kv)


# ---------- Ж-156/157: N и seed ----------
def test_n1_po_semejstvam(kuby):
    proverka_n1(kuby)


def test_seed_sobolya_v7_novye(kuby):
    for sid, k in kuby.items():
        s6, s7 = _yml(V6 / f"kub_{sid}.yaml")["sobol"], k["sobol"]
        assert {s7["seed"], s7["seed2"], s7["morris_seed"]}.isdisjoint({s6["seed"], s6["seed2"], s6["morris_seed"]}), sid
        assert len({s7["seed"], s7["seed2"], s7["morris_seed"]}) == 3 and s7["scramble"] is True, sid


# ---------- двойной фон ----------
def test_dvojnoj_fon_tochka_vybrasyvaetsya(kuby):
    proverka_dvojnoj_fon(kuby, c7.canon_v7)


def test_dvojnoj_fon_net_osi_otnositelno_u_s5_s6(kuby):
    assert "otnositelno" not in kuby["S5"]["osi7"] and "otnositelno" not in kuby["S6"]["osi7"]


def test_tochki_v7_bez_dvojnogo_fona(kuby):
    out, vyb = c7.tochki_v7(kuby["S1"], 0, 600)
    assert vyb["vybrosheno"] + len(out) == 600
    for _, c in out:
        assert c.get("k_okup", 3) == 3 and c.get("otnositelno", "sektor") != "net"
        assert not (c.get("otnositelno") and c.get("rel_signal", "syroy") != "syroy")


# ---------- SHA256SUMS ----------
def test_sha256sums_sovpadayut_s_faylami():
    assert c7.sha_nesovpadenia() == []
    tab = dict(reversed(ln.split(None, 1)) for ln in (V7 / "SHA256SUMS").read_text().splitlines() if ln.strip())
    tab = {k.strip().lstrip("*"): v for k, v in tab.items()}
    for f in V7.glob("kub_S*.yaml"):
        assert tab[f.name] == hashlib.sha256(f.read_bytes()).hexdigest(), f.name
    assert {f.name for f in V7.glob("kub_S*.yaml")} == set(tab)


def test_sha_pravka_yaml_bez_hasha_lovitsya_detektorom(tmp_path):
    d = tmp_path / "v7"
    shutil.copytree(V7, d)
    assert c7.sha_nesovpadenia(d) == []
    f = d / "kub_S1.yaml"
    f.write_text(f.read_text(encoding="utf-8").replace("N: 100000", "N: 100001"))
    assert c7.sha_nesovpadenia(d) == ["kub_S1.yaml"]
    (d / "kub_S9.yaml").write_text("x: 1\n")
    assert "kub_S9.yaml" in c7.sha_nesovpadenia(d)


@pytest.mark.xfail(strict=True, reason="ДЕФЕКТ #909: load_kub/proverit не сверяют SHA256SUMS — правка yaml без хэша загружается молча")
def test_sha_pravka_yaml_bez_hasha_otkaz_zagruzki(tmp_path):
    d = tmp_path / "v7"
    shutil.copytree(V7, d)
    f = d / "kub_S1.yaml"
    f.write_text(f.read_text(encoding="utf-8").replace("N: 100000", "N: 100001"))
    with pytest.raises(Exception):
        c7.load_kub("S1", root=d)


def test_pravka_signala_otkaz_proverit(kuby):
    k = copy.deepcopy(kuby["S1"])
    k["signal"]["z_entry"]["setka"] = [1.5, 1.75, 2.5]
    with pytest.raises(ValueError):
        c7.proverit(k)


# ---------- детерминизм ----------
_SKRIPT = ("import json,hashlib;from fastbt import canon_v7 as c;k=c.load_kub('S1');o,v=c.tochki_v7(k,0,300);"
           "s=json.dumps([[i,x] for i,x in o],sort_keys=True);print(hashlib.sha256(s.encode()).hexdigest(),len(o),v['vybrosheno'])")


def test_determinizm_dva_processa_raznyj_hashseed():
    res = []
    for hs in ("1", "987654"):
        r = subprocess.run([sys.executable, "-c", _SKRIPT], cwd=ROOT, env=dict(os.environ, PYTHONHASHSEED=hs),
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-500:]
        res.append(r.stdout.split())
    assert res[0] == res[1] and int(res[0][1]) > 100


# ---------- мутации: каждая роняет свою проверку ----------
def test_mut_k_okup_2_3_lovitsya(kuby):
    m = copy.deepcopy(kuby)
    m["S1"]["osi7"]["k_okup"]["setka"] = ["net", 2, 3]
    with pytest.raises(AssertionError):
        proverka_k_okup(m)


def test_mut_sosedi_ne_po_poryadku_yaml_lovitsya(kuby):
    m, izmenena = copy.deepcopy(kuby), 0
    for sid, k in m.items():
        for n, v in k["signal"].items():
            g = _v6_grids(sid)[n]
            if len(g) >= 4 and g[:3] != _ozhid(g):
                v["setka"], izmenena = g[:3], izmenena + 1
    assert izmenena > 0
    with pytest.raises(AssertionError):
        proverka_signal(m)


def test_mut_sosedi_otsortirovany_lovitsya(kuby):
    m, izmenena = copy.deepcopy(kuby), 0
    for sid, k in m.items():
        for n, v in k["signal"].items():
            g = _v6_grids(sid)[n]
            if len(g) >= 3 and all(isinstance(x, (int, float)) for x in g) and sorted(g) != g:
                v["setka"], izmenena = _ozhid(sorted(g)), izmenena + 1
    if izmenena == 0:
        pytest.skip("в S1–S6 нет числовых осей с неотсортированным порядком yaml: мутация неприменима")
    with pytest.raises(AssertionError):
        proverka_signal(m)


def test_mut_s6_n1_100000_lovitsya(kuby):
    m = copy.deepcopy(kuby)
    m["S6"]["sobol"]["N"] = 100_000
    with pytest.raises(AssertionError):
        proverka_n1(m)


def test_mut_dvojnoj_fon_ne_snyat_lovitsya(kuby):
    def bez_pravila(sid, pt, kub):
        kont = {a: b for a, b in kub["v6"]["kontekst"].items() if a != "rel_signal"}
        return c7.canon_v7(sid, pt, dict(kub, v6=dict(kub["v6"], kontekst=kont)))
    with pytest.raises(AssertionError):
        proverka_dvojnoj_fon(kuby, bez_pravila)


def test_mut_vykl_k_okup_ostayotsya_v_kanone_lovitsya(kuby):
    k = kuby["S5"]
    ox = c7.osi(k, c7.GOTOVY_V7)
    rng = np.random.default_rng(1)
    for _ in range(100):
        pt = dict(c7.tochka(ox, rng.random(len(ox))), k_okup="net")
        ref = c6.canon_v6("S5", {a: b for a, b in pt.items() if a != "k_okup"}, k["v6"])
        if ref is not None:
            break
    assert ref is not None
    assert dict(ref, k_okup="net") != c7.canon_v7("S5", pt, k)


def test_mut_sha_pravka_lovitsya(tmp_path):
    d = tmp_path / "v7"
    shutil.copytree(V7, d)
    f = d / "kub_S6.yaml"
    f.write_text(f.read_text(encoding="utf-8").replace("N: 50000", "N: 100000"))
    assert c7.sha_nesovpadenia(d) == ["kub_S6.yaml"]
