"""Волна E программы Ж-206: пути, запись таблиц со статусом, реестр trials ансамблей/гибридов, счёт N для Холма. Прежние реестры только читаются."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from models_v2.s6 import paths6

OUT = paths6.OUT / "e"
REP = paths6.REP
ENS = REP / "xgb_lgbm_ensemble"
HYB = {"H1": REP / "har_xgboost", "H2": REP / "har_lightgbm", "H3": REP / "har_xgb_lgbm"}
TRIALS = ENS / "OPTUNA_TRIALS.csv"
CHECK = ENS / "CHECKPOINT.json"
COLS = ("trial_id", "wave", "direction", "config", "metric_inner", "seed", "time")
LIMITS = {"ens_hybrid": 40, "E1": 15, "E2": 25}
STATUS = paths6.STATUS + "; DEVELOPMENT (выбор по внутреннему критерию crit_wf); p — сырые, множественный поиск учтён только где указан Холм (потолок)"


def write(df: pd.DataFrame, path, note: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# %s; волна E Ж-206; %s\n" % (STATUS, note))
        df.to_csv(f, index=False, lineterminator="\n")


def read(path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def tid(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:12]


def load_trials() -> pd.DataFrame:
    if TRIALS.exists():
        return pd.read_csv(TRIALS)
    return pd.DataFrame(columns=list(COLS))


def register(rows: list) -> int:
    """Дописать trials (дубль trial_id — не новый trial). Лимиты: E1 <= 15, E2 <= 25, всего <= 40. Возврат — число строк реестра."""
    t = load_trials()
    new = [r for r in rows if r["trial_id"] not in set(t["trial_id"])]
    t = pd.concat([t, pd.DataFrame(new, columns=list(COLS))], ignore_index=True) if new else t
    n = {w: int((t["wave"] == w).sum()) for w in ("E1", "E2")}
    if n["E1"] > LIMITS["E1"] or n["E2"] > LIMITS["E2"] or len(t) > LIMITS["ens_hybrid"] or t["trial_id"].duplicated().any():
        raise ValueError("бюджет волны E превышен или дубль trial_id: %s" % n)
    ENS.mkdir(parents=True, exist_ok=True)
    t.to_csv(TRIALS, index=False, lineterminator="\n")
    CHECK.write_text(json.dumps({"algorithm": "xgb_lgbm_ensemble", "limits": {"ens_hybrid": 40, "E1": 15, "E2": 25},
                                 "used": {"ens_hybrid": len(t), "E1": n["E1"], "E2": n["E2"]}, "updated": time.strftime("%Y-%m-%d %H:%M")},
                                ensure_ascii=False, indent=1))
    return len(t)


def trial(cfg: dict, wave: str, direction: str, metric: str, secs: float = 0.0) -> dict:
    return {"trial_id": tid(cfg), "wave": wave, "direction": direction, "config": json.dumps(cfg, ensure_ascii=False, sort_keys=True, default=str),
            "metric_inner": metric, "seed": 42, "time": "%s;%.1fs" % (time.strftime("%Y-%m-%d %H:%M"), secs)}


def n_trials() -> int:
    """N для Холма: уникальные trials программы Ж-206 по реестрам xgboost/, lightgbm/, xgb_lgbm_ensemble/ на момент вызова."""
    ids = set()
    for p in (REP / "xgboost" / "OPTUNA_TRIALS.csv", REP / "lightgbm" / "OPTUNA_TRIALS.csv", TRIALS):
        if p.exists():
            ids |= {(p.parent.name, i) for i in pd.read_csv(p)["trial_id"]}
    return len(ids)
