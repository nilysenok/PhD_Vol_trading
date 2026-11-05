"""Волна B (Ж-206), поток XGB: объявленные конфигурации obj/feat на базе Б-2+F4, trial_id = хэш конфигурации, дубли Ж-204, реестр trials.
Реестр только дописывается; дубль прежних запусков — не trial (ссылка на прежний id)."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from models_v2.s6 import paths6

XGB_REP = paths6.REP / "xgboost"
OLD = paths6.OLD_REP
LIMIT_OBJ_FEAT = 45
TRIAL_HEAD = ["trial_id", "wave", "direction", "config", "metric_inner", "seed", "time"]
F4 = ("F4",)


def mk(name: str, loss: str = "l2", gw: float = 0.0, groups: tuple = F4, drop: str = "", direction: str = "objective", cond: bool = False) -> dict:
    return {"name": name, "loss": loss, "gw": float(gw), "groups": tuple(groups), "drop": drop, "direction": direction, "cond": cond}


CTRL = mk("ctrl_f4", direction="control")
NEW = [mk("x3_gamma_f4", "gamma"), mk("x4_qlike_f4", "qlike"), mk("x5_l2_g05_f4", "l2", 0.5), mk("x5_l2_g10_f4", "l2", 1.0),
       mk("x5_qlike_g05_f4", "qlike", 0.5),
       mk("f4_f2", groups=("F4", "F2"), direction="feature"),
       mk("f4_abl_no_imoex", drop="no_imoex", direction="feature"), mk("f4_abl_no_market", drop="no_market", direction="feature"),
       mk("f4_abl_no_semivar", drop="no_semivar", direction="feature")]
COND = [mk("f4_f1_f2", groups=("F4", "F1", "F2"), direction="feature", cond=True)]   # вторая условная (лучший объектив × F4+F2) строится в runx.stage_cond по результату
LOSS_TARGET = {"l2": "ln(RV/S) − ln(RV_m/S)", "gamma": "RV/S", "qlike": "RV/S"}


def canon(c: dict) -> dict:
    """Каноническая конфигурация для trial_id (поля, меняющие прогноз; имя и направление не входят)."""
    return {"alg": "xgboost", "loss": c["loss"], "gw": c["gw"], "groups": list(c["groups"]), "drop": c["drop"], "target": LOSS_TARGET[c["loss"]],
            "hp": "I2 (годовой LOYO, сетка 132)", "seed": 42, "window": "expanding", "retrain": "daily", "early_stopping": "I1"}


def trial_id(c: dict) -> str:
    return hashlib.sha256(json.dumps(canon(c), sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def prior_trial_ids() -> dict:
    """{ключ конфигурации: (row_id Ж-204, trial_id)} для конфигураций, уже посчитанных в Ж-204 на базе Б-2+F4 (дубли)."""
    b2 = pd.read_csv(OLD / "OPTUNA_TRIALS_B2.csv")
    f4 = b2[b2["конфигурация"].str.contains(r"name='b2_f4'", regex=False)]["trial_id"].iloc[0]
    c = pd.read_csv(OLD / "OPTUNA_TRIALS_C.csv")
    f41 = c[c["конфигурация"].apply(lambda s: bool(re.search(r'"kind": "c1_f4_f1"', s)))]["trial_id"].iloc[0]
    return {key(mk("x", groups=("F4",))): ("b2_f4", str(f4)), key(mk("x", groups=("F4", "F1"))): ("c1_f4_f1", str(f41))}


def key(c: dict) -> tuple:
    return (c["loss"], c["gw"], tuple(sorted(c["groups"])), c["drop"])


def is_duplicate(c: dict, prior: dict | None = None) -> tuple | None:
    """(row_id Ж-204, trial_id) если конфигурация уже считалась в Ж-197…Ж-204, иначе None."""
    return (prior if prior is not None else prior_trial_ids()).get(key(c))


def register(path: Path, rows: list) -> int:
    """Дописывает только новые trial_id в OPTUNA_TRIALS.csv; возвращает число добавленных."""
    old = pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame(columns=TRIAL_HEAD)
    new = [r for r in rows if r["trial_id"] not in set(old["trial_id"])]
    if new:
        pd.concat([old, pd.DataFrame(new)], ignore_index=True)[TRIAL_HEAD].to_csv(path, index=False)
    return len(new)


def sync_checkpoint(path: Path, trials_path: Path) -> dict:
    """used.obj_feat = число строк реестра волны B (только новые уникальные); остальные счётчики не трогаются."""
    ck = json.loads(path.read_text())
    n = len(pd.read_csv(trials_path)) if trials_path.exists() and trials_path.stat().st_size else 0
    if n > LIMIT_OBJ_FEAT:
        raise ValueError("бюджет obj/feat: %d > %d" % (n, LIMIT_OBJ_FEAT))
    ck["used"]["obj_feat"] = n
    path.write_text(json.dumps(ck, ensure_ascii=False, indent=1))
    return ck
