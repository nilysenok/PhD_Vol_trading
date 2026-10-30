"""V7-1a: загрузка куба v7 сверяет sha256 с SHA256SUMS (Д5): правка yaml без хэша, чужой файл без строки — отказ."""
import hashlib
import shutil

import pytest

from fastbt import canon_v7 as c7

try:
    from fastbt import canon_v7b as c7b
except ImportError:  # ветка v7-kub-a без S8–S12
    c7b = None


def _kopiya(tmp_path, sid):
    for f in c7.KUB7.glob("kub_S*.yaml"):
        shutil.copy(f, tmp_path / f.name)
    shutil.copy(c7.KUB7 / c7.SUMS, tmp_path / c7.SUMS)
    return tmp_path / f"kub_{sid}.yaml"


def _pravka(p):
    p.write_text(p.read_text(encoding="utf-8") + "\n# правка\n", encoding="utf-8")


def test_neizmenennye_kuby_gruzyatsya():
    for sid in ("S1", "S6"):
        assert c7.load_kub(sid)["sid"] == sid


def test_pravka_yaml_bez_hasha_otkaz(tmp_path):
    _pravka(_kopiya(tmp_path, "S1"))
    with pytest.raises(ValueError, match=r"kub_S1\.yaml"):
        c7.load_kub("S1", tmp_path)


def test_otkaz_nazyvaet_oba_hasha(tmp_path):
    p = _kopiya(tmp_path, "S2")
    ozhid = next(ln.split()[0] for ln in (tmp_path / c7.SUMS).read_text().splitlines() if ln.endswith("kub_S2.yaml"))
    _pravka(p)
    fakt = hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError) as e:
        c7.load_kub("S2", tmp_path)
    assert fakt in e.value.args[0] and ozhid in e.value.args[0]


def test_net_stroki_v_sums_otkaz(tmp_path):
    _kopiya(tmp_path, "S1")
    s = tmp_path / c7.SUMS
    s.write_text("".join(ln for ln in s.read_text().splitlines(True) if "kub_S1.yaml" not in ln))
    with pytest.raises(ValueError, match=r"kub_S1\.yaml"):
        c7.load_kub("S1", tmp_path)


@pytest.mark.skipif(c7b is None, reason="canon_v7b — ветка v7-kub-b")
def test_v7b_pravka_yaml_bez_hasha_otkaz(tmp_path):
    _pravka(_kopiya(tmp_path, "S9"))
    with pytest.raises(ValueError, match=r"kub_S9\.yaml"):
        c7b.load_kub("S9", tmp_path)


@pytest.mark.skipif(c7b is None, reason="canon_v7b — ветка v7-kub-b")
def test_v7b_neizmenennyj_gruzitsya():
    assert c7b.load_kub("S9")["sid"] == "S9"
