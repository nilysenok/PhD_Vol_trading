"""V4-B: кэш прогнозов B. Ключ = хэш ВСЕХ входов (har_*, k23_*, D2) и КОДА (v4_b*.py, v4_io.py, константы) — Д5; подмена любого входа или правка кода даёт промах кэша."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from models_v2.s11 import v4_b_fit as FT, v4_io as IO

OUT = IO.P.ROOT / "data/regen/harj_hybrid_v4"
CODE_DIR = Path(__file__).resolve().parent


def input_files() -> list:
    fs = []
    for t in IO.T17:
        fs += [IO.path_of("H", t), IO.path_of("K2", t)]
    return fs + [IO.path_of("D2", "x")]


def code_files() -> list:
    return sorted(CODE_DIR.glob("v4_b*.py")) + [CODE_DIR / "v4_io.py"]


def digest(files, extra: str = "") -> str:
    h = hashlib.sha256(extra.encode())
    for f in files:
        h.update(Path(f).name.encode())
        h.update(IO.sha256(f).encode())
    return h.hexdigest()


def keys(extra_in=None, extra_code=None) -> dict:
    """-> {'inputs': хэш входов, 'code': хэш кода + гиперпараметры, 'key': общий}."""
    ins = digest(input_files() if extra_in is None else extra_in)
    code = digest(code_files() if extra_code is None else extra_code, "kappa=%s;lam=%s;min_pairs=%s" % (FT.KAPPA, FT.LAM, FT.MIN_PAIRS))
    return {"inputs": ins, "code": code, "key": hashlib.sha256((ins + code).encode()).hexdigest()}


def cached(name: str, key: str, build):
    """name: файл без расширения в OUT. build() -> dict {суффикс: DataFrame}. Возвращает {суффикс: DataFrame}; пересчёт при промахе ключа."""
    OUT.mkdir(parents=True, exist_ok=True)
    kf = OUT / "B_cache_key.json"
    ks = json.loads(kf.read_text()) if kf.exists() else {}
    files = lambda sfx: OUT / ("%s%s.parquet" % (name, sfx))
    sfx = list(ks.get(name + "__sfx", []))
    if ks.get(name) == key and sfx and all(files(s).exists() for s in sfx):
        return {s: pd.read_parquet(files(s)) for s in sfx}
    res = build()
    for s, df in res.items():
        df.to_parquet(files(s))
    ks[name], ks[name + "__sfx"] = key, list(res)
    kf.write_text(json.dumps(ks, ensure_ascii=False, indent=1))
    return res
