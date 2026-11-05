"""Реестр trials (только дописывание) и контрольная точка программы Ж-204. Волна A: 0 trials."""
from __future__ import annotations

import json

from models_v2.s5 import common

TRIAL_COLS = ("trial_id", "волна", "направление", "конфигурация", "метрика_внутр_валидации", "метрики_описания", "время", "seed", "повторный_seed")
LIMITS = {"objectives_target": 35, "feature_engineering": 45, "har_boosting": 50, "rolling_recency": 35, "regime_aware": 50, "ensemble_calibration": 25}


def ensure(cache_key: str) -> None:
    """Создаёт файлы, только если их нет; существующие не перезаписываются (дописывание — дело следующих волн)."""
    common.REP.mkdir(parents=True, exist_ok=True)
    t, c = common.REP / "OPTUNA_TRIALS.csv", common.REP / "CHECKPOINT.json"
    if not t.exists():
        t.write_text(",".join(TRIAL_COLS) + "\n", encoding="utf-8")
    if not c.exists():
        c.write_text(json.dumps({"program": "Ж-204 SBER h=1 deep", "waves": {"A": "done", "B": "pending", "C": "pending", "D": "pending"},
                                 "trials_limit": LIMITS, "trials_limit_total": sum(LIMITS.values()), "trials_used": {k: 0 for k in LIMITS},
                                 "cache_key_wave_a": cache_key}, ensure_ascii=False, indent=1))
