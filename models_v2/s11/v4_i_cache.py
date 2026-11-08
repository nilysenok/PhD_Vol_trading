"""V4-I: ключ кэша прогнозов I1–I3 и компонентного прогноза = хэш ВСЕХ входов (har_*, k23_*, D2, daily_<T>.parquet) и КОДА (v4_i_comp/fit/run/cache, v4_b_run, v4_rqc_calib, v4_io, константы) — Д5.
Кэш — `data/regen/harj_hybrid_v4/I<k>.parquet`, `I<k>_coef.parquet`, `I_comp.parquet`, `I_loto_*.parquet`, ключи — `I_cache_key.json`. `cached(write=False)` — только чтение (`--check`): промах ключа — исключение, файлы не пишутся."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from models_v2.s11 import v4_b_cache as BC, v4_i_fit as FT, v4_i_run as R, v4_io as IO, v4_rqc_calib as CAL

OUT, CODE_DIR = BC.OUT, Path(__file__).resolve().parent
NAMES = ("v4_i_comp.py", "v4_i_fit.py", "v4_i_run.py", "v4_i_cache.py", "v4_b_run.py", "v4_rqc_calib.py", "v4_io.py")
OWN = re.compile(r"^I(\d|_)")


def inputs() -> list:
    return BC.input_files() + [IO.path_of("DAILY", t) for t in IO.T17]


def keys(extra_in=None, extra_code=None) -> dict:
    ins = BC.digest(inputs() if extra_in is None else extra_in)
    const = "kappa=%s;lam=%s;min=%s;hl=%s;lead0" % (FT.KAPPA, FT.LAM, R.MIN_PAIRS, (R.IC.HL_P, R.IC.HL_M, CAL.MIN_PAIRS))
    cd = BC.digest(sorted(set([CODE_DIR / n for n in NAMES] if extra_code is None else extra_code)), const)
    return {"inputs": ins, "code": cd, "key": hashlib.sha256((ins + cd).encode()).hexdigest()}


def prior_sha() -> str:
    """Хэш всех прежних parquet программы (кроме I*): прежние прогнозы остаются побитно прежними (сверка в --check с записью при объявлении)."""
    h = hashlib.sha256()
    for f in sorted(OUT.glob("*.parquet")):
        if not OWN.match(f.name):
            h.update(f.name.encode())
            h.update(IO.sha256(f).encode())
    return h.hexdigest()


def cached(name: str, key: str, build, write: bool = True):
    """name — файл без расширения; build() -> dict {суффикс: DataFrame}. write=False: только чтение (`--check`), промах ключа -> RuntimeError."""
    kf = OUT / "I_cache_key.json"
    ks = json.loads(kf.read_text()) if kf.exists() else {}
    files = lambda sfx: OUT / ("%s%s.parquet" % (name, sfx))
    sfx = list(ks.get(name + "__sfx", []))
    if ks.get(name) == key and sfx and all(files(s).exists() for s in sfx):
        return {s: pd.read_parquet(files(s)) for s in sfx}
    if not write:
        raise RuntimeError("кэш %s не соответствует входам/коду: сначала `run_v4 --stage I` (проверка только читает)" % name)
    OUT.mkdir(parents=True, exist_ok=True)
    res = build()
    for s, df in res.items():
        df.to_parquet(files(s))
    ks[name], ks[name + "__sfx"] = key, list(res)
    kf.write_text(json.dumps(ks, ensure_ascii=False, indent=1))
    return res
