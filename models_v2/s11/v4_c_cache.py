"""V4-C: ключи кэша Z и прогнозов C = хэш ВСЕХ входов (har_*, k23_*, D2, daily_*, 10-минутные бары) и КОДА (v4_z*.py, v4_c*.py, используемые модули B, гиперпараметры) — Д5.
Файлы: data/regen/harj_hybrid_v4/Z_<T>.parquet, C<k>.parquet, C<k>_weights.parquet; ключи — Z_cache_key.json, C_cache_key.json."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from models_v2.s11 import v4_b_cache as BC, v4_b_fit as FT, v4_io as IO

OUT = BC.OUT
CODE_DIR = Path(__file__).resolve().parent
B_USED = ["v4_b_run.py", "v4_b_fit.py", "v4_b_ops.py", "v4_io.py"]


def z_inputs() -> list:
    fs = BC.input_files() + [IO.path_of("DAILY", t) for t in IO.T17] + [IO.P.bars_path(t) for t in IO.T17]
    return fs


def z_code() -> list:
    return sorted(CODE_DIR.glob("v4_z*.py")) + [CODE_DIR / "v4_io.py"]


def c_code() -> list:
    return sorted(CODE_DIR.glob("v4_c_*.py")) + z_code() + [CODE_DIR / n for n in B_USED]


def keys(kind: str, extra_in=None, extra_code=None) -> dict:
    """kind: 'Z' или 'C'. -> {'inputs', 'code', 'key'}; подмена любого входа или правка кода меняет ключ."""
    ins = BC.digest(z_inputs() if extra_in is None else extra_in)
    files = (z_code() if kind == "Z" else c_code()) if extra_code is None else extra_code
    code = BC.digest(sorted(set(files)), "kappa=%s;lam=%s;min_pairs=%s" % (FT.KAPPA, FT.LAM, FT.MIN_PAIRS))
    return {"inputs": ins, "code": code, "key": hashlib.sha256((ins + code).encode()).hexdigest()}


def cached(kind: str, name: str, key: str, build):
    """name — файл без расширения; build() -> dict {суффикс: DataFrame}. Промах ключа — пересчёт."""
    OUT.mkdir(parents=True, exist_ok=True)
    kf = OUT / ("%s_cache_key.json" % kind)
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
