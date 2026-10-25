"""Кэш артефактов на диске по хэшу входов и кода (приём 14).

Ключ = sha1(имя, хэши исходников модулей fastbt, переданные части). Файлы — npz в
data/fastbt/cache/ (в .gitignore). FASTBT_NO_CACHE=1 — выключить.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("FASTBT_CACHE", ROOT / "data" / "fastbt" / "cache"))
_CODE = None


def code_hash() -> str:
    """Хэш всех исходников пакета fastbt (кроме тестов): правка кода сбрасывает кэш."""
    global _CODE
    if _CODE is None:
        h = hashlib.sha1()
        base = Path(__file__).resolve().parent
        for p in sorted(base.rglob("*.py")):
            if "tests" in p.parts:
                continue
            h.update(p.read_bytes())
        _CODE = h.hexdigest()
    return _CODE


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
