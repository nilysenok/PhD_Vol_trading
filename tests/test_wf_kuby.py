"""Кубы v5/v6 объявляют период wf (лист wf-L8): только поле periody; без отметки владельца — отказ до чтения данных."""
import hashlib
import subprocess
from pathlib import Path

import pytest
import yaml

import fastbt.period_v2 as pv
import fastbt.progon_server as srv
import fastbt.progon_sobol as ps

KORENX = Path(__file__).resolve().parents[1]
SIDS = [f"S{i}" for i in range(1, 7)]
KUBY = [(v, s) for v in ("v5", "v6") for s in SIDS]
BAZA = "origin/kanon-v22-pereschet"


def put(v, s):
    return KORENX / "fastbt/designs" / v / f"kub_{s}.yaml"


def baza_kub(v, s):
    r = subprocess.run(["git", "show", f"{BAZA}:fastbt/designs/{v}/kub_{s}.yaml"], cwd=KORENX, capture_output=True, text=True)
    if r.returncode:
        pytest.skip(f"нет {BAZA}")
    return yaml.safe_load(r.stdout)


def rz(v, period="wf"):
    return ps.Rezhim(v, period, "seed", False, "t0d") if v == "v5" else ps.Rezhim("v6", period, "seed", False, "t0d", "sobol", None, ())


@pytest.mark.parametrize("v,s", KUBY)
def test_kub_ob_yavlyaet_wf(v, s):
    c = yaml.safe_load(put(v, s).read_text("utf-8"))
    assert c["periody"] == ["train4", "v2", "wf"]
    ps.proverit_v5(ps.load_kub(s, rz(v)), rz(v))                       # отметка владельца есть в репозитории: допуск проходит


@pytest.mark.parametrize("v,s", KUBY)
def test_diff_protiv_bazy_tolko_periody(v, s):
    novyj, staryj = yaml.safe_load(put(v, s).read_text("utf-8")), baza_kub(v, s)
    assert staryj["periody"] == ["train4", "v2"]
    for c in (novyj, staryj):
        c.pop("periody")
    if v == "v6":                                                      # baza.kub_sha — следствие нового sha куба v5, не новая ось
        assert novyj["baza"].pop("kub_sha") == hashlib.sha256(put("v5", s).read_bytes()).hexdigest()
        staryj["baza"].pop("kub_sha")
    assert novyj == staryj                                             # сетка, оси, N, канон — прежние


def test_mutaciya_wf_snyat_iz_odnogo_kuba_lovitsya(tmp_path):
    c = ps.load_kub("S3", rz("v5"))
    c = {**c, "periody": ["train4", "v2"]}
    with pytest.raises(ValueError, match="не объявлен в кубе"):
        ps.proverit_v5(c, rz("v5"))


def test_wf_bez_otmetki_otkaz_do_chteniya(tmp_path, monkeypatch):
    nuzhno = pv.proverit_chtenie
    monkeypatch.setattr(pv, "proverit_chtenie", lambda p, put=None, otmetka=tmp_path / "net.json": nuzhno(p, put, otmetka))
    chitano = []
    monkeypatch.setattr(srv, "progon", lambda *a, **k: chitano.append(1))
    with pytest.raises(pv.ZamokVladeltsa):
        srv.main(["--kub", "v5", "--period", "wf", "--strategii", "S1", "--out", str(tmp_path / "o")])
    with pytest.raises(pv.ZamokVladeltsa):
        srv.zagruzit_kub(put("v6", "S1"), "wf", rz=rz("v6"))
    assert not chitano and not (tmp_path / "o").exists()


def test_period_wf_granicy():
    p = pv.PERIODY["wf"]
    assert (p.progrev, p.y0, p.konec) == (pv.pd.Timestamp("2014-06-01"), 2015, pv.pd.Timestamp("2026-02-28"))


# --- допуск периода wf в etap1.build_inputs и progon_sobol.sreda (guards только для wf и только с отметкой) ---
import inspect
import textwrap

import numpy as np
import pandas as pd

import fastbt.etap1 as etap1


@pytest.fixture
def bez_otmetki(tmp_path, monkeypatch):
    nuzhno = pv.proverit_chtenie
    monkeypatch.setattr(pv, "proverit_chtenie", lambda p, put=None, otmetka=tmp_path / "net.json": nuzhno(p, put, otmetka))


@pytest.fixture
def zapis_build(monkeypatch):
    vyzovy = []
    monkeypatch.setattr(etap1, "_build", lambda sid, ev, end, period: vyzovy.append((end, period)) or "E1")
    return vyzovy


def test_build_inputs_wf_s_otmetkoj_prohodit(zapis_build):
    assert etap1.build_inputs("S1", period="wf") == "E1"
    assert zapis_build == [(pd.Timestamp("2026-03-01"), "wf")]


def test_build_inputs_wf_bez_otmetki_otkaz(bez_otmetki, zapis_build):
    with pytest.raises(pv.ZamokVladeltsa):
        etap1.build_inputs("S1", period="wf")
    assert not zapis_build


def test_build_inputs_train4_guard_prezhnij(zapis_build):
    assert etap1.build_inputs("S1", period="train4") == "E1"
    assert zapis_build[0][0] == etap1.END_TRAIN
    with pytest.raises(PermissionError):
        etap1.build_inputs("S1", end=pd.Timestamp("2022-01-01"), period="train4")


def sr(posledn):
    return ps.Sreda(type("E", (), {"fdr": "t0d"})(), "k", "v", np.array(["2015-01-02", posledn], dtype="datetime64[D]"), ["T"])


def otkaz(sreda_fn, period, posledn, monkeypatch):
    """True, если sreda_fn отказывает (ValueError) на рядах до posledn."""
    monkeypatch.setattr(ps, "fabrika", lambda sid: sr(posledn))
    ps._W.clear()
    try:
        sreda_fn("S1", rz("v5", period))
    except ValueError:
        return True
    finally:
        ps._W.clear()
    return False


def test_sreda_wf_s_otmetkoj_do_28_fevralya(monkeypatch):
    assert not otkaz(ps.sreda, "wf", "2026-02-27", monkeypatch)
    assert otkaz(ps.sreda, "wf", "2026-03-01", monkeypatch)


def test_sreda_wf_bez_otmetki_otkaz(bez_otmetki, monkeypatch):
    monkeypatch.setattr(ps, "fabrika", lambda sid: sr("2026-02-27"))
    ps._W.clear()
    with pytest.raises(pv.ZamokVladeltsa):
        ps.sreda("S1", rz("v5", "wf"))
    ps._W.clear()


def test_sreda_train4_guard_prezhnij(monkeypatch):
    assert otkaz(ps.sreda, "train4", "2021-06-01", monkeypatch)
    assert not otkaz(ps.sreda, "train4", "2019-12-30", monkeypatch)


def test_mutaciya_guard_snyat_dlya_vseh_periodov_lovitsya(monkeypatch):
    ish = textwrap.dedent(inspect.getsource(ps.sreda))
    kod = ish.replace('rz.period == "wf"', "True")
    assert kod != ish
    loc = {}
    exec(kod, ps.__dict__, loc)                                        # noqa: S102 — мутант; ps.sreda не перезаписан
    assert otkaz(ps.sreda, "train4", "2021-06-01", monkeypatch)       # настоящий: отказ
    assert not otkaz(loc["sreda"], "train4", "2021-06-01", monkeypatch)   # мутант пропускает 2021: тест выше его роняет
