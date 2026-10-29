"""Д5/мутация № 39: numba-кэш быстрого слоя зависит от хэша ВСЕХ модулей fastbt, и каталог задан при импорте fastbt
(не только через fastbt.s9). Без этого fast_exec.py не видел правок fast_exec_v6.py (находка #653)."""
import os
import subprocess
import sys
from pathlib import Path

from fastbt import cache

ROOT = Path(cache.__file__).resolve().parent


def _njit_cache_files():
    return {p for p in ROOT.rglob("*.py") if "tests" not in p.parts and "cache=True" in p.read_text(encoding="utf-8")}


def test_import_fastbt_sets_hashed_numba_dir_without_s9():
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_CACHE_DIR"}
    code = "import os, sys, fastbt; assert 'fastbt.s9' not in sys.modules; print(os.environ.get('NUMBA_CACHE_DIR',''))"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT.parent, env=env, capture_output=True, text=True, check=True)
    assert Path(out.stdout.strip()).name == cache.code_hash()[:16]


def test_hash_covers_every_njit_cache_module():
    assert _njit_cache_files() <= set(cache.hashed_files())
    for name in ("fast_exec.py", "fast_exec_v6.py", "fast_etap1.py", "fast_filtry_v5.py"):
        assert ROOT / name in set(cache.hashed_files())


def test_mutation_v6_not_in_hash_caught(monkeypatch):
    monkeypatch.setattr(cache, "hashed_files", lambda: [p for p in sorted(ROOT.rglob("*.py")) if p.name != "fast_exec_v6.py"])
    assert not (_njit_cache_files() <= set(cache.hashed_files()))


def test_edit_of_v6_changes_numba_dir(tmp_path, monkeypatch):
    (tmp_path / "cache.py").write_text("", encoding="utf-8")
    v6 = tmp_path / "fast_exec_v6.py"
    v6.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(cache, "__file__", str(tmp_path / "cache.py"))
    monkeypatch.setattr(cache, "_CODE", None)
    d1 = cache.numba_cache_dir()
    v6.write_text("x = 2\n", encoding="utf-8")
    monkeypatch.setattr(cache, "_CODE", None)
    assert cache.numba_cache_dir() != d1
