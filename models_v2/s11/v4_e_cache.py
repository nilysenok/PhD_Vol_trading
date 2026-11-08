"""V4-E: ключ кэша прогнозов E = хэш ВСЕХ входов (har_*, k23_*, D2, B_B2.parquet — база G) и КОДА (v4_e*.py, v4_io.py, константы) — Д5. Кэш: C-кэш (`CC.cached`) с видом «E»: E<k>.parquet, E_cache_key.json."""
from __future__ import annotations

from pathlib import Path

from models_v2.s11 import v4_b_cache as BC, v4_c_cache as CC, v4_e_calib as CAL, v4_e_e6 as E6, v4_io as IO

OUT, CODE_DIR = BC.OUT, Path(__file__).resolve().parent


def inputs() -> list:
    return BC.input_files() + [OUT / "B_B2.parquet"]


def code() -> list:
    return sorted(CODE_DIR.glob("v4_e_*.py")) + [CODE_DIR / "v4_io.py"]


def keys(extra_in=None, extra_code=None) -> dict:
    import hashlib
    ins = BC.digest(inputs() if extra_in is None else extra_in)
    cd = BC.digest(sorted(set(code() if extra_code is None else extra_code)), "kappa=%s;min_pairs=%s;clip=%s;e6=%s" % (CAL.KAPPA, CAL.MIN_PAIRS, CAL.B_CLIP, (E6.HL, E6.KAPPA, E6.LAM)))
    return {"inputs": ins, "code": cd, "key": hashlib.sha256((ins + cd).encode()).hexdigest()}


def cached(name: str, key: str, build):
    return CC.cached("E", name, key, build)
