# -*- coding: utf-8 -*-
"""Уборка numba-кэша на временном каталоге. Мутация «удаляет текущий хэш» ловится."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

_sp = importlib.util.spec_from_file_location(
    "ubrat", Path(__file__).resolve().parents[1] / "scripts" / "ubrat_numba_kesh.py")
U = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(U)


def _mk(tmp: Path, k: int = 8) -> Path:
    kesh = tmp / "numba_cache"
    for i in range(k):
        d = kesh / f"h{i}"
        d.mkdir(parents=True)
        (d / "f.bin").write_bytes(b"x" * 100)
        os.utime(d, (1000 + i, 1000 + i))
    return kesh


def _names(kesh: Path) -> set[str]:
    return {p.name for p in kesh.iterdir()}


def test_dry_run_nichego_ne_udalyaet(tmp_path):
    kesh = _mk(tmp_path)
    lishnie, byte = U.ubrat(kesh, "h0", 2, da=False)
    assert len(_names(kesh)) == 8 and len(lishnie) == 5 and byte == 500


def test_tekushchiy_i_n_poslednih_ostayutsya(tmp_path):
    kesh = _mk(tmp_path)
    U.ubrat(kesh, "h0", 2, da=True)  # h0 самый старый, но текущий
    assert _names(kesh) == {"h0", "h7", "h6"}


def test_put_vne_kataloga_otkaz(tmp_path):
    kesh = _mk(tmp_path)
    vne = tmp_path / "vne"
    vne.mkdir()
    (vne / "vazhno").write_text("1")
    os.symlink(vne, kesh / "ssylka")
    os.utime(kesh / "ssylka", (1, 1), follow_symlinks=False)
    with pytest.raises(ValueError):
        U.ubrat(kesh, "h7", 1, da=True)
    assert (vne / "vazhno").exists() and len(_names(kesh)) == 9
    with pytest.raises(ValueError):
        U.ubrat(tmp_path / "net_takogo", "h7", 1, da=True)


def test_mutaciya_udalyaet_tekushchiy_lovitsya(tmp_path):
    """Мутация: текущий хэш не защищён (None) — старый текущий стёрт; проверка «h0 остался» её ловит."""
    mut = _mk(tmp_path / "mut")
    U.ubrat(mut, None, 2, da=True)
    assert "h0" not in _names(mut)  # мутант виден
    ok = _mk(tmp_path / "ok")
    U.ubrat(ok, "h0", 2, da=True)
    assert "h0" in _names(ok)
