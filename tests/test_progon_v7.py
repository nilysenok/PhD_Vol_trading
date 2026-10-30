"""V7-8a: драйвер куба v7 (fastbt.progon_v7) на синтетике — данных и сети нет. Ledger 2.2 с setka v7, чекпоинт 10 000, --prodolzhit без дублей,
период polnaya (окно метрик 2017+), пилотный замер; мутации «чекпоинт не пишется» и «дубль индекса при продолжении» ловятся."""
import contextlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fastbt.ledger_v2 as lm
import fastbt.period_v2 as pv
import fastbt.progon_morris_v7 as pm7
import fastbt.progon_server as srv
import fastbt.progon_sobol as ps
import fastbt.progon_v7 as p7
from test_progon_server import Fake, Sboy, chp, syrye  # noqa: F401

SID, T = "S1", 8
DATY = np.arange(np.datetime64("2017-01-09"), np.datetime64("2017-01-09") + T)      # окно метрик polnaya: 2017+


class Fake7(Fake):
    fdr = None

    def __init__(self, plohie=()):
        self.plohie = set(plohie)


def tochki_fn(kub, n0, n1, rz):
    return [(i, {"a": float(i)}) for i in range(n0, n1)]


def pt_json(i):
    return json.dumps({"a": float(i)}, sort_keys=True)


@pytest.fixture
def sred(monkeypatch, tmp_path):
    """Синтетическая среда: куб в tmp, оценщик Fake7, без зоны wf и отметки владельца, без проверки диска."""
    (tmp_path / "kub_S1.yaml").write_text("sid: S1\n", encoding="utf-8")
    monkeypatch.setattr(p7, "KUB7", tmp_path / "kub_{sid}.yaml")
    monkeypatch.setattr(ps, "_W", {})
    monkeypatch.setattr(pv,"proverit_chtenie", lambda *a, **k: None)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(10**13, 0, 10**13))
    st = {"ev": Fake7()}
    monkeypatch.setattr(ps, "fabrika", lambda sid: ps.Sreda(st["ev"], "kod-fake", "vhody-fake", DATY, [f"T{i:02d}" for i in range(17)]))
    return st


def kubs(tmp_path, n=300):
    c = {"sid": SID, "setka": "v7", "fdr": None, "data": "2026-10-31", "sobol": {"N": n, "seed": 1, "seed2": 2}}
    return {SID: (tmp_path / "kub_S1.yaml", c)}


def gon(tmp_path, n, out=None, **kw):
    kw.setdefault("shag", 100)
    return p7.progon7(kubs(tmp_path, n), out or tmp_path / "out", 2, {SID: n}, tochki_fn=tochki_fn, rezhim="prezhniy", **kw)


def indeksy(out):
    return sorted(r["sobol_i"] for r in syrye(out, SID))


def proverit_chekpoint(out, n, shag):
    ch = chp(out, SID)
    assert ch["posledniy_i"] == n - 1 and ch["shag"] == shag, ch


# ---- ledger 2.2, setka v7 ----

def test_ledger_v7_setka_v7_rol_sobol_i_ne_smeshivaetsya_s_v6(sred, tmp_path):
    out = tmp_path / "out"
    gon(tmp_path, 120, out)
    mf = json.loads((out / SID / "manifest.json").read_text(encoding="utf-8"))
    assert (mf["v"], mf["setka"], mf["period"], mf["rol"]) == (lm.V22, "v7", "polnaya", "sobol")
    rows = syrye(out, SID)
    assert {r["setka"] for r in rows} == {"v7"} and {r["rol"] for r in rows} == {"sobol"}
    with pytest.raises(ValueError, match="setka"):                       # строка v7 в manifest v6 (и наоборот) — отказ
        lm.proverit_stroku_22(rows[0], "v6")
    with pytest.raises(ValueError, match="другой"):
        lm.Pisatel22(out, SID, 9, {**mf, "setka": "v6"}, daty=DATY)


def test_manifest_okno_polnoj_istorii(sred, tmp_path):
    out = tmp_path / "out"
    gon(tmp_path, 10, out)
    mf = json.loads((out / SID / "manifest.json").read_text(encoding="utf-8"))
    assert (mf["okno_s"], mf["progrev_s"]) == ("2017-01-01", "2014-08-26")


# ---- чекпоинт ----

def test_chekpoint_po_umolchaniyu_10000(sred, tmp_path):
    assert p7.SHAG == 10_000 and srv.CHEKPOINT == 5000                   # v7 — 10 000; прежний драйвер не тронут
    out = tmp_path / "out"
    gon(tmp_path, 300, out, shag=p7.SHAG)
    proverit_chekpoint(out, 300, 10_000)
    assert srv.CHEKPOINT == 5000                                         # подмена снята
    with p7.podmeny({}, p7.SHAG):
        z = srv.zadachi({SID: 0}, {SID: 25_000}, {SID: 25_000}, "prezhniy")
    assert {a for _, a, _, _ in z} >= {0, 10_000, 20_000} and all(a // 10_000 == (b - 1) // 10_000 for _, a, b, _ in z)


def test_chekpoint_kazhdye_shag_tochek(sred, tmp_path):
    out = tmp_path / "out"
    gon(tmp_path, 250, out, shag=100)
    proverit_chekpoint(out, 250, 100)


def test_mutatsiya_chekpoint_ne_pishetsya_lovitsya(sred, tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "pisat_chekpoint", lambda *a, **k: None)
    gon(tmp_path, 250, tmp_path / "out")
    with pytest.raises((AssertionError, FileNotFoundError)):
        proverit_chekpoint(tmp_path / "out", 250, 100)


# ---- --prodolzhit ----

def obryv(sred, tmp_path):
    """Кусок [100, 200) падает на точке 120 (BaseException), [200, 300) успевает — после обрыва в ledger дыра."""
    sred["ev"] = Fake7({pt_json(120)})
    out = tmp_path / "out"
    itog = gon(tmp_path, 300, out)
    assert itog[SID]["sboy"] and chp(out, SID)["posledniy_i"] == 99
    sred["ev"] = Fake7()
    return out


def test_prodolzhit_dosschityvaet_nedostayuschie_bez_dublej(sred, tmp_path):
    out = obryv(sred, tmp_path)
    assert indeksy(out) != list(range(300))
    gon(tmp_path, 300, out, prodolzhit=True)
    assert indeksy(out) == list(range(300))
    proverit_chekpoint(out, 300, 100)


def test_prodolzhit_bez_flaga_otkaz(sred, tmp_path):
    gon(tmp_path, 100, tmp_path / "out")
    with pytest.raises(SystemExit, match="prodolzhit"):
        gon(tmp_path, 200, tmp_path / "out")


def test_mutatsiya_dubl_indeksa_pri_prodolzhenii_lovitsya(sred, tmp_path, monkeypatch):
    out = obryv(sred, tmp_path)
    monkeypatch.setattr(ps, "zakrytye", lambda root, sid: set())          # мутация: закрытые шарды забыты
    monkeypatch.setattr(lm.Pisatel, "done", lambda self: set())
    gon(tmp_path, 300, out, prodolzhit=True)
    idx = indeksy(out)
    assert idx != list(range(300)) and len(idx) != len(set(idx))          # тест «без дублей» упал бы


# ---- период polnaya ----

def test_period_polnaya_okno_2017_progrev_ne_v_metrikah():
    p = pv.PERIODY["polnaya"]
    assert (p.progrev, p.y0, p.konec) == (pd.Timestamp("2014-08-26"), 2017, pd.Timestamp("2026-02-27"))
    assert pv.gody("polnaya") == tuple(range(2017, 2027)) and len(pv.kvartaly_spisok("polnaya")) == 37
    assert lm.gody_perioda("polnaya")[0] == 2017
    with pytest.raises(ValueError):                                       # прогрев (2016) в суммы сегментов не входит
        lm.segment_kvartala(np.array(["2016-12-30"], "datetime64[D]"), lm.gody_perioda("polnaya"))
    assert lm.segment_kvartala(np.array(["2017-01-02"], "datetime64[D]"), lm.gody_perioda("polnaya"))[0] == 0
    assert pv.PERIODY["wf"].y0 == 2015                                    # wf не тронут


def test_sreda_polnaya_otkaz_dat_posle_konca(sred, tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "fabrika", lambda sid: ps.Sreda(Fake7(), "k", "v", np.array(["2026-02-28"], "datetime64[D]"), []))
    itog = gon(tmp_path, 10, tmp_path / "out")
    assert "2026-02-28" in itog[SID]["sboy"]


def test_zona_polnaya_eto_wf(monkeypatch):
    zv = []
    from fastbt import zona
    monkeypatch.setattr(zona, "zone_ctx", lambda z: zv.append(z) or contextlib.nullcontext())
    p7.zona_ctx("polnaya"), p7.zona_ctx("train4")
    assert zv == ["wf"]


# ---- пилот, к3, Моррис ----

def test_pilot_pishet_zamer(sred, tmp_path):
    out = tmp_path / "out"
    res = p7.pilot(kubs(tmp_path, 120), out, 2, {SID: 120}, tochki_fn=tochki_fn, rezhim="prezhniy", shag=100)
    z = json.loads((out / p7.ZAMER).read_text(encoding="utf-8"))
    assert z == json.loads(json.dumps(res)) and z[SID]["n"] == 120 and z[SID]["strok"] == 120
    assert z[SID]["ledger_bajt"] > 0 and z[SID]["sek_na_tochku_sredn"] > 0 and z[SID]["rss_pik_mb"] > 0
    assert z[SID]["ledger_bajt_na_1000"] == int(z[SID]["ledger_bajt"] * 1000 / 120)


def test_k3_ssylka_na_v6(tmp_path):
    v6 = tmp_path / "v6"
    (v6 / SID).mkdir(parents=True)
    (v6 / SID / "manifest.json").write_text("{}", encoding="utf-8")
    z = json.loads(p7.k3_ssylka(tmp_path / "o", v6, [SID]).read_text(encoding="utf-8"))
    assert z[SID]["put"] == str(v6 / SID) and len(z[SID]["manifest_sha"]) == 64
    with pytest.raises(SystemExit, match="к3 v6 не посчитан"):
        p7.k3_ssylka(tmp_path / "o", v6, ["S2"])


def test_kub_i_morris_bez_canon_v7_otkaz(monkeypatch):
    monkeypatch.setattr(p7.importlib, "import_module", lambda n: (_ for _ in ()).throw(ModuleNotFoundError(name=n)))
    with pytest.raises(SystemExit, match="canon_v7"):
        p7.zagruzit_kubi([SID], 1)


def test_morris_podmena_canon_vozvrashchaetsya():
    staryj, fake = pm7.pm.canon_v6, object()
    with pm7.kanon_v7(fake):
        assert pm7.pm.canon_v6 is fake and pm7.pol.canon_v6 is fake
    assert pm7.pm.canon_v6 is staryj and pm7.pol.canon_v6 is staryj


def test_prezhnie_podmeny_snyaty(sred, tmp_path):
    ps_staryj = (ps.kub_put, ps.tochki, ps.manifest, ps.sreda)
    gon(tmp_path, 10, tmp_path / "out")
    assert (ps.kub_put, ps.tochki, ps.manifest, ps.sreda) == ps_staryj
