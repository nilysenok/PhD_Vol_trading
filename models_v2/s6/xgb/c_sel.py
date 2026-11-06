"""Волна C: критерий для Optuna (J — только годы 2017–2021), оси с эффектом (cap), годовой PIT-выбор X2 (GATES_C_xgb Q3, Q4). Только фолды y < Y."""
from __future__ import annotations

import hashlib
import json

import pandas as pd

from models_v2.s6.xgb import c_space, critx

GUIDE_YEARS = tuple(range(2017, 2022))
TOL = 1e-12


def pen_by_year(folds: pd.DataFrame) -> dict:
    """{Y: pen_Y}: pen_Y = mean_y q + 0,25·(max_y q − mean_y q), y < Y (фолды y >= Y отбрасывает critx)."""
    return critx.crit(folds)["by_year"]


def j_guide(folds: pd.DataFrame) -> float:
    """Цель Optuna: среднее pen_Y только по Y = 2017…2021 (цели < 01.01.2021)."""
    by = pen_by_year(folds)
    return float(sum(by[Y] for Y in GUIDE_YEARS) / len(GUIDE_YEARS))


def effect_axes(deltas: dict, cap: int = c_space.CAP_AXES) -> list:
    """deltas {ось: min Δcrit_pen по двум точкам}; ось с эффектом: Δ < −1e-12; не больше cap осей с наименьшим Δ."""
    eff = sorted((a for a, d in deltas.items() if d < -TOL), key=lambda a: (deltas[a], a))
    return eff[:cap]


def annual_select(base: pd.DataFrame, cands: dict) -> tuple:
    """base — фолды И2-базы; cands {метка: фолды}. -> ({Y: метка | None}, составные фолды). None = И2. Ничья |Δ| <= 1e-12 — И2; иначе первая в порядке cands."""
    pens = {None: pen_by_year(base), **{k: pen_by_year(v) for k, v in cands.items()}}
    frames = {None: base, **cands}
    choice = {}
    for Y in sorted(pens[None]):
        lo = min(p[Y] for p in pens.values())
        tied = [k for k, p in pens.items() if p[Y] - lo <= TOL]
        choice[Y] = None if None in tied else tied[0]
    comp = pd.concat([frames[choice[Y]][frames[choice[Y]]["Y"] == Y] for Y in sorted(choice)], ignore_index=True)
    return choice, comp


def choice_id(family: str, choice: dict) -> str:
    """Хэш годового выбора (метка = trial_id кандидата или 'I2')."""
    s = json.dumps({"family": family, "choice": {str(Y): (v or "I2") for Y, v in sorted(choice.items())}}, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:12]
