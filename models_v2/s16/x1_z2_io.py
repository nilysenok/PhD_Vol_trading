"""Z2: пути, CHECKPOINT (слияние ключей `Z2*`, чужие сохраняются), хэш кода и CSV-шапка, хэш ЧУЖИХ выходов (для `--check`: до и после равен)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from models_v2.s16 import x1_reg as R, x1_ya_stage as SA, x1_yb_stage as SB

ROOT, REP = R.ROOT, R.REP
OUT, ZOUT, CP_FILE, REPORT = ROOT / "data/regen/harj_x1/Z2", ROOT / "data/regen/harj_x1/Z", REP / "CHECKPOINT.json", REP / "Z2_REPORT.md"
HEAD = "# диагностика/оценка, не отбор; DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; после просмотра Y1–Y4, ZD, Z; Ж-246 п. 6, этап Z2 (H1, Hc; SBER, h = 1); код z2 %s\n"


def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("x1_z2_*.py")) + [Path(R.__file__)]:
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def csv_text(df: pd.DataFrame) -> str:
    return HEAD % code_hash() + df.to_csv(index=False, lineterminator="\n")


def read_cp() -> dict:
    return json.loads(CP_FILE.read_text(encoding="utf-8")) if CP_FILE.exists() else {}


def merge_cp(key: str, val: dict) -> None:
    """Слияние: чужие ключи сохраняются."""
    cp = read_cp()
    cp[key] = val
    CP_FILE.write_text(json.dumps(cp, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")


def foreign_hash() -> str:
    """Хэш ЧУЖИХ выходов: WATCH, Z, Y-A, Y-B и docs/otchety/harj_x1 без файлов Z2_* и ключей Z2* CHECKPOINT."""
    h = hashlib.sha256(SA.digest(SA.WATCH + [ZOUT, SA.OUT, SB.OUT]).encode())
    for f in sorted(p for p in REP.rglob("*") if p.is_file() and not p.name.startswith("Z2_")):
        if f.name == "CHECKPOINT.json":
            h.update(str(f).encode() + json.dumps({k: v for k, v in json.loads(f.read_text(encoding="utf-8")).items() if not k.startswith("Z2")}, sort_keys=True).encode())
        else:
            h.update(str(f).encode() + SA.sha_file(f).encode())
    return h.hexdigest()


def read_table(name: str) -> pd.DataFrame:
    return pd.read_csv(REP / (name + ".csv"), comment="#")
