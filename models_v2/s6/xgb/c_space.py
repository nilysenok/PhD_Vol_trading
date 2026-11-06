"""Волна C (Ж-206), поток XGB: оси гиперпараметров, сдвиг вокруг годового выбора И2, trial_id. Объявлено в GATES_C_xgb.md (Q2, Q3) до счёта.
u — число шагов оси: mul — v·step^u, add — v + u·step; результат зажимается в границы (число зажатых лет публикуется)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from models_v2.s5.c.fitx import HPX

# ось: (поле HPC, тип шага, шаг, (нижняя, верхняя) граница значения, точки X1 в u, диапазон u для Optuna)
AXES = {
    "max_depth": ("depth", "add", 1.0, (1, 6), (-1, 1), (-2, 2)),
    "min_child_weight": ("min_child", "mul", 2.0, (2.0, 200.0), (-1, 1), (-2, 2)),
    "learning_rate": ("lr", "mul", 1.5, (0.01, 0.2), (-1, 1), (-2, 2)),
    "subsample": ("subsample", "add", 0.1, (0.4, 1.0), (-1, 1), (-2, 2)),
    "colsample_bytree": ("colsample", "add", 0.2, (0.3, 1.0), (-1, -2), (-3, 0)),
    "reg_alpha": ("alpha", "add", 1.0, (0.0, 5.0), (1, 3), (0, 5)),
    "reg_lambda": ("l2", "mul", 3.0, (0.05, 200.0), (-1, 1), (-2, 2)),
    "gamma": ("gamma", "add", 0.05, (0.0, 1.0), (1, 4), (0, 6)),
    "max_delta_step": ("mds", "add", 0.1, (0.0, 5.0), (1, 3), (0, 5)),
}
CAP_AXES = 4
TOL = 1e-12


@dataclass(frozen=True)
class HPC(HPX):
    """HPX (depth, min_child, l2, lr, subsample, colsample, max_trees, gamma, alpha) + max_delta_step (0 = выключено)."""
    mds: float = 0.0


def to_hpc(h) -> HPC:
    return HPC(int(h.depth), float(h.min_child), float(h.l2), float(h.lr), float(h.subsample), float(h.colsample), int(h.max_trees),
               float(getattr(h, "gamma", 0.0)), float(getattr(h, "alpha", 0.0)), float(getattr(h, "mds", 0.0)))


def shift_hp(hp: HPC, axis: str, u: float) -> HPC:
    """Сдвиг одной оси на u шагов от HP года; зажим в границы; глубина — целая."""
    fld, kind, st, (lo, hi), _, _ = AXES[axis]
    v = getattr(hp, fld)
    nv = v * st ** u if kind == "mul" else v + u * st
    nv = min(max(nv, lo), hi)
    nv = int(round(nv)) if fld == "depth" else float(nv)
    return HPC(**{**hp.__dict__, fld: nv})


def apply(base: dict, moves: dict) -> dict:
    """{Y: HPC} со всеми сдвигами moves {ось: u}; порядок осей — по имени (детерминированно)."""
    out = {}
    for Y, h in base.items():
        for ax in sorted(moves):
            h = shift_hp(h, ax, moves[ax])
        out[Y] = h
    return out


def years_changed(base: dict, hpy: dict) -> int:
    return sum(1 for Y in base if base[Y] != hpy[Y])


def clamped_years(base: dict, axis: str, u: float) -> int:
    """В скольких годах точка зажата границей (значение отличается от несжатого v·step^u / v+u·step)."""
    fld, kind, st = AXES[axis][:3]
    n = 0
    for h in base.values():
        v = getattr(h, fld)
        raw = v * st ** u if kind == "mul" else v + u * st
        n += int(abs(getattr(shift_hp(h, axis, u), fld) - raw) > 1e-12 * max(1.0, abs(raw)))
    return n


def x1_points() -> list:
    """Объявленные 18 точек: [(ось, u)] в порядке AXES."""
    return [(ax, float(u)) for ax, spec in AXES.items() for u in spec[4]]


def canon(family: str, moves: dict, kind: str = "hp") -> dict:
    """Каноническая конфигурация (поля, меняющие прогноз): семья (l2 / gamma), сдвиги, источник HP."""
    return {"alg": "xgboost", "wave": "C", "kind": kind, "family": family, "features": "Б-2+F4", "moves": {k: float(moves[k]) for k in sorted(moves)},
            "hp": "I2 (годовой LOYO, сетка 132) + сдвиг", "seed": 42, "window": "expanding", "retrain": "daily", "early_stopping": "I1"}


def trial_id(family: str, moves: dict, kind: str = "hp") -> str:
    return hashlib.sha256(json.dumps(canon(family, moves, kind), sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def name_of(family: str, moves: dict) -> str:
    return family + "_" + "_".join("%s%+g" % (a, moves[a]) for a in sorted(moves)) if moves else family + "_base"


def neighbors(moves: dict) -> list:
    """Соседи ±1 шаг по каждой оси сдвига в пределах диапазона u; u=0 — опора (не соседний расчёт)."""
    out = []
    for ax in sorted(moves):
        lo, hi = AXES[ax][5]
        for d in (-1, 1):
            u = moves[ax] + d
            if lo - TOL <= u <= hi + TOL:
                m = {**moves, ax: float(u)}
                out.append({k: v for k, v in m.items() if abs(v) > TOL})
    return out
