"""V5-0: пути, шапки таблиц, хэши входов и кода (Д5), общий кадр оценки (контроли + 46 рядов V4 + M). Только чтение входов; запись — `write_csv`/`write_text` (вызывает стадия, не `--check`)."""
from __future__ import annotations

import functools
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s10 import hp_paths as P
from models_v2.s11 import v4_fin_data as FD, v4_io as IO4

ROOT = P.ROOT
REP = ROOT / "docs/otchety/harj_hybrid_v5"
OUT = ROOT / "data/regen/harj_hybrid_v5"
STATE = ROOT / "state/harj_hybrid_v5.md"
V4REP = IO4.REP
CODE_VERSION = "v5.0"
STATUS = "DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-226, лист 0 (диагностика, моделей не обучали)"
OUTPUTS = ["CONTROLS.csv", "CRIT_AUDIT.csv", "CRIT_EQY.csv", "PASS_RV.csv", "PASS_SCORE.csv", "PASS_JUMP.csv", "PASS_SPECTRAL.csv", "SEARCH_HISTORY.csv"]
T17 = list(P.MAIN17)
M16 = list(P.M16)


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@functools.lru_cache(maxsize=1)
def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("*.py")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


@functools.lru_cache(maxsize=1)
def inputs_hash() -> str:
    """Хэш входов: INVENTORY V4 (SHA файлов прогнозов), все parquet V4, бары 17 бумаг."""
    h = hashlib.sha256(sha_file(V4REP / "INVENTORY.csv").encode() + FD.parquet_sha().encode())
    for t in T17:
        h.update(sha_file(P.bars_path(t)).encode())
    return h.hexdigest()[:16]


def header(note: str) -> str:
    return "# %s; код %s %s; входы %s; %s\n" % (STATUS, CODE_VERSION, code_hash(), inputs_hash(), note)


def write_csv(df: pd.DataFrame, name: str, note: str) -> Path:
    REP.mkdir(parents=True, exist_ok=True)
    p = REP / name
    with open(p, "w", encoding="utf-8") as f:
        f.write(header(note))
        df.to_csv(f, index=False, lineterminator="\n")
    return p


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(REP / name, comment="#")


def outputs_digest(extra=()) -> str:
    """Суммарный хэш выходов V5 (таблицы, отчёты, CHECKPOINT, кэш): для «до и после» у `--check`."""
    h = hashlib.sha256()
    files = [f for d in (REP, OUT) if d.exists() for f in sorted(d.rglob("*")) if f.is_file()] + [STATE] + list(extra)
    for f in files:
        if f.exists():
            h.update(str(f).encode() + sha_file(f).encode())
    return h.hexdigest()


@functools.lru_cache(maxsize=1)
def frame() -> pd.DataFrame:
    """Общий кадр (бумага, дата): rv_tplus, t_plus, group, контроли k1,k2,k3,d2, 46 рядов V4 и M = sqrt(K2·K3). Копия не отдаётся: вызывающий не меняет."""
    _, k = FD.frame()
    k = k.copy()
    k["M"] = np.sqrt(k["k2"] * k["k3"])
    return k


CTRL5 = ["k1", "k2", "k3", "d2", "M"]
CFGS = list(FD.CFGS)
