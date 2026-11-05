"""Волна A программы Ж-206: пути, чтение, sha256, мир Б-2 (F, сетка, дизайн, I2). Только чтение прежних данных."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2 import prov
from models_v2.s5 import common

ROOT = prov.ROOT
OUT = ROOT / "data/regen/xgb_lgbm_full_v1"
REP = ROOT / "docs/otchety/xgb_lgbm_full_v1"
OLD_DATA = (ROOT / "data/regen/v2_sber_h1", ROOT / "data/regen/v2_sber_h1_deep_v1")
OLD_REP = ROOT / "docs/otchety/sber_h1_deep_v1"
N_GRID = 2271
CONTROL_DATES = ("2018-06-01", "2022-06-01", "2025-06-02")
YEARS = tuple(range(2017, 2027))
STATUS = "HISTORICAL WALK-FORWARD (2017–02.2026, уже просмотрено; не независимый OOS)"


def sha_file(path) -> str:
    return common.sha_file(path)


def rel(path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


GATES_COMMIT = "c28e584e"   # GATES_A.md до счёта
FIX_COMMIT = "a697209a"     # ↺ 06.11: правки волны D Ж-204 после ревью (другая сессия, 00:13) — легитимно новее GATES_A


def ref_time(commit: str = GATES_COMMIT) -> float:
    """Время коммита (граница «прежнее не тронуто»): по умолчанию — коммит GATES_A.md."""
    import subprocess
    out = subprocess.run(["git", "show", "-s", "--format=%ct", commit], capture_output=True, text=True, cwd=ROOT).stdout.split()
    return float(out[-1])


def mkdirs() -> None:
    for a in ("xgboost", "lightgbm"):
        (OUT / a).mkdir(parents=True, exist_ok=True)
        (REP / a).mkdir(parents=True, exist_ok=True)


def read_csv(path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def write_csv(df: pd.DataFrame, path, note: str) -> None:
    """Шапка-комментарий со статусом (A-правила: статус в каждой таблице)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# %s; волна A Ж-206; %s; trials бюджета 0 (контроли)\n" % (STATUS, note))
        df.to_csv(f, index=False)


def dump_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def world() -> dict:
    """Мир Б-2 без записи в прежние каталоги: F, сетка, дизайн m_mkx (dz0), I2 по годам (сетка 132)."""
    from models_v2 import run_fix
    F, grid, DZ, _, _ = run_fix.prepare(str(common.IMOEX))
    return {"F": F, "grid": np.asarray(grid, int), "dz0": DZ["m_mkx"], "i2": run_fix.load_i2("i2_choice_g132")}


def controls() -> pd.DataFrame:
    return common.controls()


def sigma_frame(fc: pd.DataFrame) -> pd.DataFrame:
    """Приведение к виду d, target_date, rv, sigma2 с проверкой сетки (2271 дата, σ̂² > 0, конечна)."""
    if len(fc) != N_GRID or not (np.isfinite(fc["sigma2"]).all() and (fc["sigma2"] > 0).all()):
        raise ValueError("сетка/σ̂²: %d строк" % len(fc))
    return fc[["d", "target_date", "rv", "sigma2"]].reset_index(drop=True)
