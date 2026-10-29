"""Р-7: запись кэша s9 и указателя атомарна (уникальный tmp + os.replace)."""
from pathlib import Path

import pytest

from fastbt import s9


def _dvuhpriemnaya(vstavka):
    def zap(self, data):
        with open(self, "wb" if isinstance(data, bytes) else "w") as f:
            f.write(data[: len(data) // 2])
            f.flush()
            vstavka()
            f.write(data[len(data) // 2:])
    return zap


@pytest.mark.parametrize("data", [b"A" * 1000, "B" * 1000])
def test_chitatel_v_pauze_vidit_net_ili_celyy(tmp_path, monkeypatch, data):
    tgt = tmp_path / "cache_x.pkl"
    vidno = []
    zap = _dvuhpriemnaya(lambda: vidno.append(tgt.read_bytes() if tgt.exists() else None))
    monkeypatch.setattr(Path, "write_bytes", zap)
    monkeypatch.setattr(Path, "write_text", zap)
    s9._atomic_write(tgt, data)
    assert vidno == [None]
    assert tgt.read_bytes() == (data if isinstance(data, bytes) else data.encode())
    assert [f.name for f in tmp_path.iterdir()] == ["cache_x.pkl"]


def test_dva_pisatelya_v_pauze_ne_portyat_drug_druga(tmp_path, monkeypatch):
    tgt = tmp_path / "c.pkl"
    pervyy, vtoroy = b"1" * 1000, b"2" * 1000
    voshli = []

    def vlozhennyy():
        if not voshli:
            voshli.append(1)
            s9._atomic_write(tgt, vtoroy)
    monkeypatch.setattr(Path, "write_bytes", _dvuhpriemnaya(vlozhennyy))
    s9._atomic_write(tgt, pervyy)
    assert tgt.read_bytes() in (pervyy, vtoroy)
    assert sorted(f.name for f in tmp_path.iterdir()) == ["c.pkl"]


def test_build_cache_pishet_kesh_i_ukazatel_cherez_atomarnuyu(tmp_path, monkeypatch):
    vyzovy = []
    monkeypatch.setattr(s9, "_atomic_write", lambda p, d: vyzovy.append(p.name))
    monkeypatch.setattr(s9, "OUT", tmp_path)
    monkeypatch.setattr(s9, "PERIOD", "train4")
    monkeypatch.setattr(s9, "_job", lambda a: (a[0], {}))
    monkeypatch.setattr(s9, "cache_key", lambda: "K")
    s9.build_cache(workers=1)
    assert vyzovy == ["cache_K.pkl", s9._latest_name()]
