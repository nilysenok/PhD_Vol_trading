"""Волна D (Ж-206), поток XGB: конфигурации окон / весов / частоты / режимных признаков, trial_id, HP баз, спецификации (GATES_D_xgb Q1–Q7).
Базы: f4 = ctrl_f4 (Б-2+F4, годовой И2), j = x2_best_j (И2 + сдвиг X2 из X2_DECISION.json; колонки 2017–2021 — потолок)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

import numpy as np

from models_v2.s4 import rows
from models_v2.s5.b1 import designs, jobs
from models_v2.s5.b2 import engine
from models_v2.s6.xgb import c_calc, c_space, cfgx, worldx

# имя окна: (rolling N пар | 0, полураспад весов в строках | 0); 252 строки = 1 год
WINDOWS = {"expanding": (0, 0), "roll504": (504, 0), "roll756": (756, 0), "roll1008": (1008, 0), "roll1260": (1260, 0), "exp756": (0, 756), "exp1260": (0, 1260)}
D1_WINDOWS = tuple(k for k in WINDOWS if k != "expanding")
BASES = ("j", "f4")
LIMIT = 30
FREQ_THRESHOLD = 0.0005
FREQS = ("d", "w", "m")


@dataclass(frozen=True)
class DCfg:
    base: str
    win: str = "expanding"
    freq: str = "d"
    r1: str = ""            # '' | 'qw' | 'all'


def name_of(c: DCfg) -> str:
    return "%s_%s_%s%s" % (c.base, c.win, c.freq, "_r1" + c.r1 if c.r1 else "")


def canon(c: DCfg) -> dict:
    """Поля, меняющие прогноз (имя не входит): база, окно, веса, частота, режимные признаки, HP, seed."""
    n, hl = WINDOWS[c.win]
    return {"alg": "xgboost", "wave": "D", "base": c.base, "features": "Б-2+F4", "r1": c.r1, "window_pairs": n, "half_life_rows": hl, "retrain": c.freq,
            "hp": "I2 (годовой LOYO, сетка 132)" + (" + сдвиг X2 (x2_best_j)" if c.base == "j" else ""), "seed": 42, "early_stopping": "I1"}


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def trial_id(c: DCfg) -> str:
    return digest(canon(c))


def hpy_of(w: dict, base: str) -> dict:
    """{Y: HPC}: ctrl_f4 — годовой выбор И2; x2_best_j — он же со сдвигами лучшей точки X2 (чтение решения C)."""
    b = c_calc.hpc_base(w)
    if base == "f4":
        return b
    dec = json.loads((cfgx.XGB_REP / "X2_DECISION.json").read_text())
    return c_space.apply(b, dec["best_moves"])


def spec_for(w: dict, r1: str = ""):
    """Spec Б-2+F4 (ln-цель); r1: + q_w | + q_w, stress_RV, переход (`designs.regime_cols`, только прошлое: 252 предыдущих значения)."""
    s = worldx.spec_of(w, cfgx.CTRL)
    if not r1:
        return s
    rg = designs.regime_cols(w["F"])
    cols = [rg["qw"]] if r1 == "qw" else [rg["qw"], rg["stress"], rg["trans"]]
    return dataclasses.replace(s, name=s.name + "_r1" + r1, X=np.column_stack([s.X] + cols))


def window_idx(idx: np.ndarray, window: int) -> np.ndarray:
    """Последние `window` пар окна (0 — все); пары идут по возрастанию t."""
    return idx[-window:] if window else idx


def pair_counts(F, spec, grid) -> np.ndarray:
    """Число доступных пар на дате d сетки (t+1 <= d)."""
    return np.array([len(jobs.pairs(F, spec, int(d))) for d in grid])


def availability(F, spec, grid) -> list:
    """[(год цели, окно, доля дат года, где пар >= N)] для rolling-окон; для весов — не нужно (все пары)."""
    n, yr = pair_counts(F, spec, grid), np.array([rows.year_of(F, int(d)) for d in grid])
    return [(int(y), k, float((n[yr == y] >= WINDOWS[k][0]).mean())) for k in WINDOWS if WINDOWS[k][0] for y in sorted(set(yr))]


def blocks(F, grid, freq: str) -> dict:
    return engine.blocks(F, grid, freq)


def default_cfgs() -> list:
    """D1: 6 окон × 2 базы = 12 конфигураций (trials)."""
    return [DCfg(b, k) for b in BASES for k in D1_WINDOWS]
