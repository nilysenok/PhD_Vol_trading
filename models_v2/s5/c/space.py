"""C1: пространство Optuna и соседи ±1 шаг (GATES_C §2). Границы объявлены до счёта."""
from __future__ import annotations

import hashlib
import json

from models_v2.s5.c.fitx import HPX

# ось: (нижняя, верхняя, тип шага, шаг); 'mul' — умножение/деление, 'add' — сложение/вычитание
AXES = {"depth": (1, 3, "add", 1), "min_child": (10.0, 200.0, "mul", 2.0), "l2": (0.5, 100.0, "mul", 3.0), "lr": (0.02, 0.10, "mul", 1.5),
        "subsample": (0.5, 1.0, "add", 0.1), "colsample": (0.3, 1.0, "add", 0.1), "gamma": (0.0, 0.05, "add", 0.01),
        "alpha": (0.0, 5.0, "add", 1.0)}
MAX_TREES, STUDY_SEED, N_STARTUP, N_TPE = 300, 42, 8, 22
GUIDE_YEARS = tuple(range(2017, 2022))   # годы Y, по которым считается J для TPE
ALL_YEARS = tuple(range(2017, 2027))


def suggest(trial) -> HPX:
    s = lambda n: trial.suggest_float(n, AXES[n][0], AXES[n][1], log=AXES[n][2] == "mul" and n != "gamma")  # noqa: E731
    return HPX(trial.suggest_int("depth", 1, 3), s("min_child"), s("l2"), s("lr"), s("subsample"), s("colsample"), MAX_TREES,
               trial.suggest_float("gamma", 0.0, 0.05), trial.suggest_float("alpha", 0.0, 5.0))


def in_bounds(hp, axis: str) -> bool:
    lo, hi = AXES[axis][:2]
    v = getattr(hp, axis)
    return lo - 1e-12 <= v <= hi + 1e-12


def shift(hp, axis: str, sign: int):
    """Сдвиг одной оси на один шаг (sign = ±1); None, если выходит за границу."""
    lo, hi, kind, st = AXES[axis]
    v = getattr(hp, axis)
    nv = v * st ** sign if kind == "mul" else v + sign * st
    if axis == "depth":
        nv = int(round(nv))
    if nv < lo - 1e-12 or nv > hi + 1e-12:
        return None
    d = {f: getattr(hp, f) for f in ("depth", "min_child", "l2", "lr", "subsample", "colsample", "max_trees", "gamma", "alpha") if hasattr(hp, f)}
    d.update({"gamma": getattr(hp, "gamma", 0.0), "alpha": getattr(hp, "alpha", 0.0)})
    d[axis] = nv
    return HPX(**d)


def neighbors_by_year(hp_y: dict) -> dict:
    """{(ось, ±1): {Y: HPX}}; сосед строится, только если сдвиг допустим во все годы."""
    out = {}
    for ax in AXES:
        for sg in (-1, 1):
            cand = {Y: shift(h, ax, sg) for Y, h in hp_y.items()}
            if all(v is not None for v in cand.values()):
                out[(ax, sg)] = cand
    return out


def hp_dict(hp) -> dict:
    return {a: round(float(getattr(hp, a, 0.0)), 10) for a in AXES} | {"max_trees": getattr(hp, "max_trees", MAX_TREES)}


def trial_id(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:12]
