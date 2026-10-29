"""Кэш артефактов на диске по хэшу входов и кода (приём 14).

Ключ = sha1(имя, хэши исходников модулей fastbt, переданные части). Файлы — npz в
<основная копия>/data/fastbt/cache/ (О5). FASTBT_NO_CACHE=1 — выключить.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# О5: данные и артефакты — только в общем data/ основной копии (не в копии агента)
MAIN = Path(os.environ.get("FASTBT_MAIN", "/Users/nilysenok/Desktop/Старое/PhD_Vol_trading"))
CACHE = Path(os.environ.get("FASTBT_CACHE", MAIN / "data" / "fastbt" / "cache"))
_CODE = None


def hashed_files() -> list[Path]:
    """Исходники пакета fastbt (кроме тестов), входящие в хэш кода, — в т.ч. все модули с njit(cache=True)."""
    base = Path(__file__).resolve().parent
    return [p for p in sorted(base.rglob("*.py")) if "tests" not in p.parts]


def code_hash() -> str:
    """Хэш всех исходников пакета fastbt (кроме тестов): правка кода сбрасывает кэш."""
    global _CODE
    if _CODE is None:
        h = hashlib.sha1()
        for p in hashed_files():
            h.update(p.read_bytes())
        _CODE = h.hexdigest()
    return _CODE


def numba_cache_dir() -> Path:
    """Д5: каталог numba-кэша зависит от хэша кода fastbt/*.py — правка любого модуля (в т.ч. fast_exec_v6) строит кэш заново."""
    return MAIN / "data" / "fastbt" / "numba_cache" / code_hash()[:16]


def key(name: str, *parts) -> str:
    h = hashlib.sha1(name.encode())
    h.update(code_hash().encode())
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(np.ascontiguousarray(p).tobytes())
        else:
            h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return f"{name}_{h.hexdigest()[:20]}"


def enabled() -> bool:
    return os.environ.get("FASTBT_NO_CACHE", "") != "1"


def load(k: str):
    p = CACHE / f"{k}.npz"
    if not enabled() or not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        return {n: z[n] for n in z.files}


def save(k: str, arrays: dict) -> None:
    if not enabled():
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = CACHE / f"{k}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, CACHE / f"{k}.npz")
