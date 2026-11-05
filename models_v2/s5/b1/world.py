"""Мир B1: F, сетка, DZ (формы Б-2 / остаток / уровень), HP по годам И2, PIT-граница ε. Кэш с ключом входов и кода (Д5)."""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass

import numpy as np

from models_v2 import prov
from models_v2.s4 import prov4
from models_v2.s5 import common, pit

B1_DIR = common.DEEP / "b1"
PKG = common.ROOT / "models_v2"


@dataclass
class World:
    F: object
    grid: np.ndarray
    DZ: dict
    lb: np.ndarray
    i2: dict
    eps: np.ndarray


def code_files() -> list:
    fs = sorted((PKG / "s5/b1").glob("*.py")) + sorted((PKG / "s2").glob("*.py")) + sorted((PKG / "s3").glob("*.py"))
    fs += sorted((PKG / "s4").glob("*.py")) + [PKG / "s5/pit.py", PKG / "s5/common.py", PKG / "run_fix.py"]
    return fs


def code_sha() -> str:
    h = hashlib.sha256(prov.code_sha().encode())
    for f in code_files():
        h.update(common.sha_file(f).encode())
    return h.hexdigest()[:16]


def inputs_sha() -> str:
    parts = [common.sha_file(prov.BARS), common.sha_file(common.IMOEX), prov4.sha_papers(),
             common.sha_file(common.TAB / "fix_i2_choice_g132.csv"), common.sha_file(common.OUT / "fix_forecasts_c.parquet"),
             common.sha_file(common.OUT / "forecasts_p1.parquet"), common.sha_file(common.OUT / "stress_p3.parquet")]
    return hashlib.sha256("".join(parts).encode()).hexdigest()[:16]


def key() -> str:
    return code_sha() + "-" + inputs_sha()


def load(force: bool = False) -> World:
    B1_DIR.mkdir(parents=True, exist_ok=True)
    p, k = B1_DIR / "world.pkl", B1_DIR / "world.key"
    if not force and p.exists() and k.exists() and k.read_text() == key():
        return pickle.loads(p.read_bytes())
    from models_v2 import run_fix   # тяжёлый импорт — только при пересчёте
    F, grid, DZ, _, _ = run_fix.prepare(str(common.IMOEX))
    w = World(F, grid, DZ, DZ["resid_mkx"].off, run_fix.load_i2("i2_choice_g132"), pit.eps_floor(F.rv))
    p.write_bytes(pickle.dumps(w))
    k.write_text(key())
    return w
