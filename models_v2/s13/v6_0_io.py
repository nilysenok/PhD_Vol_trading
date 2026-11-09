"""V6-0 (Ж-227): пути, шапки таблиц, хэши входов и кода (Д5), суммарный хэш выходов. Только SBER, h = 1. Запись — `write_csv`/`write_text` (вызывает стадия, не `--check`)."""
from __future__ import annotations

import functools
import hashlib
from pathlib import Path

import pandas as pd

from models_v2.s10 import hp_paths as P

ROOT = P.ROOT
REP = ROOT / "docs/otchety/harj_hybrid_v6"
OUT = ROOT / "data/regen/harj_hybrid_v6"
STATE = ROOT / "state/harj_hybrid_v6.md"
V5REP = ROOT / "docs/otchety/harj_hybrid_v5"
V5OUT = ROOT / "data/regen/harj_hybrid_v5"
V5STATE = ROOT / "state/harj_hybrid_v5.md"
CODE_VERSION = "v6.0"
STATUS = "DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-227, лист 0 (SBER, h = 1; аудит и паспорт, моделей не обучали)"
CAVEAT = "K2/K3 SBER: дизайн выбирался на SBER 2017–2021 (sber_h1_deep_v1; Ж-204/206/210), оговорка для всех сравнений"
TABLES = ["CONTROLS_SBER.csv", "PRIOR_SBER.csv", "BEST_SBER.csv", "PASS_M10_SBER.csv", "RESID_SBER.csv", "CRIT_SBER.csv"]
HEADER_TRIALS = "key,program,source,direction,config,kind\n"


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@functools.lru_cache(maxsize=1)
def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("v6_0_*.py")):          # только код листа 0: листы 1+ (v6_1_*.py) не меняют хэш выходов листа 0
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


@functools.lru_cache(maxsize=1)
def inputs_hash() -> str:
    """Хэш входов: SHA всех файлов, из которых читаются ряды SBER (реестр `v6_0_load`), и баров SBER."""
    from models_v2.s13 import v6_0_load as LD
    h = hashlib.sha256()
    for f in sorted(set(LD.source_files()) | {P.bars_path("SBER")}):
        h.update(str(f.relative_to(ROOT)).encode() + sha_file(f).encode())
    return h.hexdigest()[:16]


def header(note: str) -> str:
    return "# %s; код %s %s; входы %s; %s; %s\n" % (STATUS, CODE_VERSION, code_hash(), inputs_hash(), CAVEAT, note)


def write_csv(df: pd.DataFrame, name: str, note: str) -> Path:
    REP.mkdir(parents=True, exist_ok=True)
    p = REP / name
    with open(p, "w", encoding="utf-8") as f:
        f.write(header(note))
        df.to_csv(f, index=False, lineterminator="\n")
    return p


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(REP / name, comment="#")


def _digest(dirs, extra) -> str:
    h = hashlib.sha256()
    files = [f for d in dirs if d.exists() for f in sorted(d.rglob("*")) if f.is_file()] + list(extra)
    for f in files:
        if f.exists():
            h.update(str(f).encode() + sha_file(f).encode())
    return h.hexdigest()


def outputs_digest() -> str:
    """Суммарный хэш выходов V6 (таблицы, отчёты, CHECKPOINT, кэш, state): для «до и после» у `--check`."""
    return _digest((REP, OUT), (STATE,))


def v5_digest() -> str:
    """Суммарный хэш выходов V5 (G9): таблицы, кэш, state."""
    return _digest((V5REP, V5OUT), (V5STATE,))
