"""V4-RQC: ключ кэша прогнозов K8, RQC1–RQC3 = хэш ВСЕХ входов (har_*, k23_*, D2, Z_<T>.parquet — доля скачка и Δ ln RV) и КОДА (v4_rqc_calib/r3/run, v4_io, константы) — Д5.
Кэш — C-кэш (`CC.cached`) с видом «RQC»: K8.parquet, RQC<k>.parquet, RQC_cache_key.json. Отчёт и проверки в ключ не входят."""
from __future__ import annotations

import hashlib
from pathlib import Path

from models_v2.s11 import v4_b_cache as BC, v4_c_cache as CC, v4_io as IO, v4_rqc_calib as CAL, v4_rqc_r3 as R3

OUT, CODE_DIR = BC.OUT, Path(__file__).resolve().parent
NAMES = ("v4_rqc_calib.py", "v4_rqc_r3.py", "v4_rqc_run.py")


def inputs() -> list:
    return BC.input_files() + [OUT / ("Z_%s.parquet" % t) for t in IO.T17]


def code() -> list:
    return [CODE_DIR / n for n in NAMES] + [CODE_DIR / "v4_io.py"]


def keys(extra_in=None, extra_code=None) -> dict:
    ins = BC.digest(inputs() if extra_in is None else extra_in)
    const = "a=[%s,%s];min=%s;r3=%s" % (CAL.A_LO, CAL.A_HI, CAL.MIN_PAIRS, (R3.HL, R3.KAPPA, R3.STEP, R3.FREEZE, R3.MIN_PAIRS, R3.Q_LO, R3.Q_HI, R3.Q_JS, R3.Q_DL))
    cd = BC.digest(sorted(set(code() if extra_code is None else extra_code)), const)
    return {"inputs": ins, "code": cd, "key": hashlib.sha256((ins + cd).encode()).hexdigest()}


def prior_sha() -> str:
    """Хэш всех прежних parquet программы (кроме K8, RQC*): прежние прогнозы должны остаться побитно прежними (сверка в --check с записью при объявлении)."""
    h = hashlib.sha256()
    for f in sorted(OUT.glob("*.parquet")):
        if not f.name.startswith(("K8", "RQC", "ZH_", "H_")) and not (f.name[:1] == "H" and f.name[1:2].isdigit()):       # отступление (лист H): файлы ZH_*, H1–H8*, H_* — выходы этапа H, не «прежние»
            h.update(f.name.encode())
            h.update(IO.sha256(f).encode())
    return h.hexdigest()


def cached(name: str, key: str, build):
    return CC.cached("RQC", name, key, build)
