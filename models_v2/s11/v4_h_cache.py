"""V4-H: ключ кэша признаков ZH_<T>.parquet и прогнозов H1–H8 = хэш ВСЕХ входов (har_*, k23_*, D2, Z_<T>.parquet, daily_<T>.parquet) и КОДА (v4_h_feat/fit/run/cache, v4_cld_brk, v4_c_fit, v4_b_run, v4_rqc_calib, v4_io, константы) — Д5.
Кэш — файлы `data/regen/harj_hybrid_v4/ZH_<T>.parquet`, `H<k>.parquet`, `H<k>_coef.parquet`, ключи — `H_cache_key.json`. `cached(write=False)` — только чтение (режим `--check`): промах ключа — исключение, файлы не пишутся."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from models_v2.s11 import v4_b_cache as BC, v4_h_fit as HFIT, v4_h_feat as HF, v4_io as IO, v4_rqc_calib as CAL

OUT, CODE_DIR = BC.OUT, Path(__file__).resolve().parent
NAMES = ("v4_h_feat.py", "v4_h_fit.py", "v4_h_run.py", "v4_h_cache.py", "v4_cld_brk.py", "v4_c_fit.py", "v4_b_run.py", "v4_rqc_calib.py", "v4_io.py")
OWN = re.compile(r"^(ZH_|H[1-8](_|\.)|H_)")


def inputs() -> list:
    return BC.input_files() + [OUT / ("Z_%s.parquet" % t) for t in IO.T17] + [IO.path_of("DAILY", t) for t in IO.T17]


def keys(extra_in=None, extra_code=None) -> dict:
    ins = BC.digest(inputs() if extra_in is None else extra_in)
    const = "kappa=%s;lam_proj=%s;clip=%s;min=%s;feat=%s" % (HFIT.KAPPA, HFIT.LAM_PROJ, HFIT.CLIP, 60, (HF.HL, HF.WIN, HF.MINW, HF.Q_JUMP, CAL.MIN_PAIRS))
    cd = BC.digest(sorted(set([CODE_DIR / n for n in NAMES] if extra_code is None else extra_code)), const)
    return {"inputs": ins, "code": cd, "key": hashlib.sha256((ins + cd).encode()).hexdigest()}


def prior_sha() -> str:
    """Хэш всех прежних parquet программы (кроме ZH_*, H1–H8*, H_*): прежние прогнозы, Z и BRK остаются побитно прежними (сверка в --check с записью при объявлении)."""
    h = hashlib.sha256()
    for f in sorted(OUT.glob("*.parquet")):
        if not OWN.match(f.name):
            h.update(f.name.encode())
            h.update(IO.sha256(f).encode())
    return h.hexdigest()


def cached(name: str, key: str, build, write: bool = True):
    """name — файл без расширения; build() -> dict {суффикс: DataFrame}. write=False: только чтение (`--check`), промах ключа -> RuntimeError."""
    kf = OUT / "H_cache_key.json"
    ks = json.loads(kf.read_text()) if kf.exists() else {}
    files = lambda sfx: OUT / ("%s%s.parquet" % (name, sfx))
    sfx = list(ks.get(name + "__sfx", []))
    if ks.get(name) == key and sfx and all(files(s).exists() for s in sfx):
        return {s: pd.read_parquet(files(s)) for s in sfx}
    if not write:
        raise RuntimeError("кэш %s не соответствует входам/коду: сначала `run_v4 --stage H` (проверка только читает)" % name)
    OUT.mkdir(parents=True, exist_ok=True)
    res = build()
    for s, df in res.items():
        df.to_parquet(files(s))
    ks[name], ks[name + "__sfx"] = key, list(res)
    kf.write_text(json.dumps(ks, ensure_ascii=False, indent=1))
    return res
