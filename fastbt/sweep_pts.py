"""Точки сетки для режимов Э1-в (Ж-031, толкование 27.10 19:35): уровни, сведение неактивных параметров, плато, параметры с эффектом.

Вынесено из `sweep_run_iter` (лимит 200 строк). Расчёта нет: только преобразование точек и таблиц; `sweep_drivers` владельца не правится.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from fastbt import sweep_drivers as sd
from fastbt.sweep_run import METR

RADIUS, MIN_NB, TOR_MIN = 1, 3, 0.05
# при выходах в √RV эти ключи не доходят до расчёта (rm_sqrt_rv.split/apply берёт k_sl, k_trail, k_be, okno_rv, max_hold, cooldown_bars, trail_n)
INERT_SQRT = ("sl_mult", "tp_mult", "breakeven_trigger", "time_decay", "initial_sl_mult", "trail_type", "trail_atr_mult",
              "breakeven_thresh")


def levels(vals: list, v) -> list:
    """Три узла: база в центре, на краю сетки — три ближайших узла; сетка из двух узлов — оба узла, без центра."""
    if len(vals) < 2 or v not in vals:
        raise ValueError(f"факторный 3 уровня: в сетке {vals} нужно ≥ 2 узлов и база {v!r}")
    if len(vals) == 2:
        return list(vals)
    j = min(max(vals.index(v), 1), len(vals) - 2)
    return vals[j - 1:j + 2]


def make_fix(exp: dict):
    """Точка -> каноническая: параметры условного блока при невыполненном условии и ATR-ключи при выходах √RV — в базу."""
    base, usl = exp["base"], exp["uslovie"]

    def fix(p: dict) -> dict:
        q = dict(p)
        if "sqrt_rv" in (q.get("rm_edinitsa"), q.get("rm_tip")):
            q.update({k: base[k] for k in INERT_SQRT if k in q and k in base})
        for k, cond in usl.items():
            if k in q and any(q.get(ck) != cv for ck, cv in cond.items()):
                q[k] = base[k]
        return q
    return fix


def effect_params(exp: dict, out: Path, name: str, c0: float, thr: float) -> list:
    """Параметры сетки с размахом цели ≥ thr в `S{k}_oat_tornado_{c0}.csv`; нет файла — отказ (пространство не угадываем)."""
    f = Path(out) / f"{name}_oat_tornado_{c0:g}.csv"
    t = pd.read_csv(f, comment="#")
    keep = set(t.loc[t["objective_range"] >= thr, "param"])
    return [k for k in exp["grid"] if k in keep]


def optuna_limits(sw, exp: dict, o: dict, per: tuple, out: Path, name: str, c0: float) -> dict:
    """Ограничения Optuna: `space` — параметры с эффектом (если задан tornado_min), `min_sd` — предел активности (сделок/год)."""
    d = {}
    if o.get("tornado_min") is not None:
        d["space"] = effect_params(exp, out, name, c0, o["tornado_min"])
        if not d["space"]:
            raise ValueError(f"{name}: нет параметров с размахом ≥ {o['tornado_min']} в торнадо — Optuna искать негде")
    if o.get("min_frac"):
        d["min_sd"] = o["min_frac"] * sw.evaluate(exp["base"], per[0], c0)["sdelok_y"]
    return d


def plateau_sparse(rows: list, keys: list, grid: dict) -> list:
    """Плато Optuna: среднее цели по посчитанным точкам в ±RADIUS узлов сетки по каждому ключу (с самой точкой, как у модуля)."""
    ix = [[grid[k].index(r[k]) if r[k] in grid[k] else -99 for k in keys] for r in rows]
    for i, r in enumerate(rows):
        nb = [o["objective"] for j, o in enumerate(rows)
              if all(abs(a - b) <= RADIUS for a, b in zip(ix[i], ix[j])) and o["objective"] == o["objective"]]
        r.update(plateau=sum(nb) / len(nb) if nb else float("nan"), neighbors=len(nb), edge=len(nb) < MIN_NB)
    return sorted(rows, key=lambda r: -(r["plateau"] if r["plateau"] == r["plateau"] else -1e9))


def point_cols(sw, p: dict, per: tuple, c: float) -> dict:
    t, k = sw.evaluate(p, per[0], c), sw.evaluate(p, per[1], c)
    d = dict(objective=sd.objective_stable(t), sr_d=t["sr_d"], T=t["T"], skew=t["skew"], kurt=t["kurt"],
             **{f"tune_{m}": t[m] for m in METR}, tune_sharpe_2018=t["by_year"][2018])
    return dict(d, **{f"ctrl_{m}": k[m] for m in METR}, ctrl_sharpe_2019=k["by_year"][2019])
