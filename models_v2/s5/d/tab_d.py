"""D: строки таблиц, бюджет и реестр trials волны D (направление ensemble_calibration, лимит 25 по GATES_D §1 и п. 26 блока)."""
from __future__ import annotations

import json
import time

import pandas as pd

from models_v2.s5 import common, registry
from models_v2.s5.d import ens

DIRECTION = "ensemble_calibration"
COMPONENTS = ("d_lgbm_f4", "d_xgb_f4_r1008", "d_xgb_f4_hl504")
D_TRIALS = COMPONENTS + ens.TRIALS          # 3 + 13 = 16
LIM_D = 25
FAMILY = 90 + len(D_TRIALS)                  # 106: все уникальные trials программы
FAMILY_REF = FAMILY + 10                     # справочно: + 10 контролей «простое среднее»
TRIALS_D = common.REP / "OPTUNA_TRIALS_D.csv"
TRIALS_ALL = common.REP / "OPTUNA_TRIALS.csv"


def check_budget(rows: list, used_before: int) -> None:
    """Стоп при превышении лимита волны D или лимита п. 26 (25), при дубле trial_id."""
    ids = [r["trial_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("стоп: дубль trial_id в реестре D")
    if len(rows) > LIM_D or used_before + len(rows) > registry.LIMITS[DIRECTION]:
        raise SystemExit("стоп: бюджет trials ensemble_calibration превышен (%d)" % len(rows))


def trial_row(tid: str, cfg: dict, inner: str, desc: str, seed: int = 42, repeat: str = "") -> dict:
    return {"trial_id": tid, "волна": "D", "направление": DIRECTION, "конфигурация": json.dumps(cfg, ensure_ascii=False, sort_keys=True),
            "метрика_внутр_валидации": inner, "метрики_описания": desc, "время": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": seed, "повторный_seed": repeat}


def merge_registry(d_rows: list) -> dict:
    """OPTUNA_TRIALS.csv = (B1 + B2 + C без изменений) + D; id уникальны; только дописывание (повторный вызов не дублирует). -> счётчики по направлениям."""
    cols = list(registry.TRIAL_COLS)
    old = pd.read_csv(TRIALS_ALL, dtype=str).fillna("")
    old = old[old["волна"] != "D"]
    check_budget(d_rows, 0)
    new = pd.DataFrame(d_rows).astype(str).fillna("")
    allr = pd.concat([old[cols], new[cols]], ignore_index=True)
    if allr["trial_id"].duplicated().any():
        raise SystemExit("стоп: дубль trial_id в сведённом реестре")
    new.to_csv(TRIALS_D, index=False)
    allr.to_csv(TRIALS_ALL, index=False)
    return allr["направление"].value_counts().to_dict()


def write_checkpoint(counts: dict) -> None:
    p = common.REP / "CHECKPOINT.json"
    ck = json.loads(p.read_text())
    for k in registry.LIMITS:
        ck["trials_used"][k] = int(counts.get(k, 0))
    ck["waves"]["D"] = "done"
    assert sum(ck["trials_used"].values()) <= ck["trials_limit_total"]
    p.write_text(json.dumps(ck, ensure_ascii=False, indent=1))
