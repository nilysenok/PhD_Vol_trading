"""V7-0 (Ж-232): пути, шапки таблиц, хэши входов и кода (Д5), суммарный хэш выходов (для «до = после» у `--check`). Только SBER, h = 1. Запись — `write_*` (вызывает стадия, не `--check`)."""
from __future__ import annotations

import functools
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "docs/otchety/harj_hybrid_v7"
OUT = ROOT / "data/regen/harj_hybrid_v7"
STATE = ROOT / "state/harj_hybrid_v7.md"
CMP = ROOT / "docs/otchety/harj_hybrid_v6/cmp"
PAN = ROOT / "data/regen/harj_panel_hybrid"
BARS = ROOT / "data/iss_check/bars_10m_full/SBER.parquet"
DAILY = PAN / "daily_SBER.parquet"
CODE_VERSION = "v7.0"
STATUS = "DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-232, лист 0 (SBER, h = 1; аудит, паспорт M10, входы; моделей не обучали, конфигураций 0)"
CAVEAT = "K2/K3 SBER: дизайн выбирался на SBER 2017–2021 (sber_h1_deep_v1; Ж-204/206/210), оговорка для всех сравнений"


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@functools.lru_cache(maxsize=1)
def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("v7_0_*.py")) + [Path(__file__).parent / "run_v7.py"]:
        if f.exists():
            h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


@functools.lru_cache(maxsize=1)
def inputs_hash() -> str:
    """Хэш входов: бары SBER, канонические дни, K1/K2/K3 и таблицы сравнения V1–V6."""
    h = hashlib.sha256()
    for f in (BARS, DAILY, PAN / "har_SBER.parquet", PAN / "k23_SBER.parquet", CMP / "CMP_SERIES.csv", CMP / "CMP_WINDOWS.csv"):
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


def outputs_digest() -> str:
    """Суммарный хэш выходов листа 0 (отчёты, кэш, state): у `--check` до и после обязан совпасть."""
    h = hashlib.sha256()
    files = [f for d in (REP, OUT) if d.exists() for f in sorted(d.rglob("*")) if f.is_file()] + [STATE]
    for f in files:
        if f.exists():
            h.update(str(f).encode() + sha_file(f).encode())
    return h.hexdigest()
