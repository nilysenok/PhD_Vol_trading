"""DE0: пути, шапки таблиц, запись/чтение CSV, хэши. Запись — только `write_csv` / `write_text` (их зовёт стадия без `--check`)."""
from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "docs/otchety/harj_deep"
V6 = ROOT / "docs/otchety/harj_hybrid_v6"
V7 = ROOT / "docs/otchety/harj_hybrid_v7"
CMP = V6 / "cmp"
CP_FILE = REP / "CHECKPOINT.json"
DOC = REP / "DE0_REPORT.md"
CODE_VERSION = "de0"
STATUS = "диагностика, не отбор; DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-239, лист DE0 (SBER, h = 1; сохранённые прогнозы, обучения и новых конфигураций нет)"
TABLES = ["DE0_REGISTRY", "DE0_GAP_YEARS", "DE0_GAP_BASE", "DE0_UNDER_OVER", "DE0_TAILS", "DE0_CAUSES", "DE0_GROUPS", "DE0_QUESTIONS", "DE0_ACF", "DE0_RUNS", "DE0_RIDGE"]


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@functools.lru_cache(maxsize=1)
def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("de0_*.py")):      # только код DE0: листы E1… (de_*.py, run_de.py) хэш выходов DE0 не меняют
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def header(note: str = "") -> str:
    return "# %s; код %s %s; %s\n" % (STATUS, CODE_VERSION, code_hash(), note)


def csv_text(df: pd.DataFrame, note: str = "") -> str:
    return header(note) + df.to_csv(index=False, lineterminator="\n")


def write_csv(df: pd.DataFrame, name: str, note: str = "") -> None:
    REP.mkdir(parents=True, exist_ok=True)
    (REP / (name + ".csv")).write_text(csv_text(df, note), encoding="utf-8")


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(REP / (name + ".csv"), comment="#")


def read_cp() -> dict:
    return json.loads(CP_FILE.read_text(encoding="utf-8")) if CP_FILE.exists() else {}


def merge_checkpoint(key: str, value: dict) -> None:
    """Слияние ключей: чужие ключи (других листов) сохраняются (класс № 117)."""
    cp = read_cp()
    cp[key] = value
    CP_FILE.write_text(json.dumps(cp, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")


def outputs_hash(root: Path | None = None) -> str:
    """Суммарный хэш выходов DE0 (CSV, отчёт); CHECKPOINT исключён — он хранит хэши таблиц."""
    h = hashlib.sha256()
    for f in sorted((root or REP).glob("DE0_*")):
        h.update(f.name.encode() + sha_file(f).encode())
    return h.hexdigest()
