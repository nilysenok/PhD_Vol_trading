"""C1: Optuna TPE (seed 42) по Б-2+F4. Цель Optuna J(θ) — среднее score_Y по Y 2017–2021 (цели < 01.01.2021); выбор HP_Y — по score_Y (PIT)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna

from models_v2.s4 import select
from models_v2.s5.c import engine_c, fitx, space


def J(sc: dict, years=space.GUIDE_YEARS) -> float:
    """Целевая функция TPE: только годы Y из years (score_Y использует цели < 01.01.Y)."""
    return float(np.mean([sc[Y] for Y in years]))


def choose(cand: dict, i2: dict, Y: int):
    """cand: {k: (hp, {Y: score})}; i2: {Y: (hp, score)}. HP_Y = argmin score_Y по {k} ∪ {i2}; ничья |Δ| <= 1e-12 — select.order_key.
    Использует только score_Y (не score_{Y'})."""
    pool = {("tpe", k): (h, sc[Y]) for k, (h, sc) in cand.items()}
    pool[("i2", Y)] = (i2[Y][0], i2[Y][1])
    ok = {k: v[1] for k, v in pool.items() if np.isfinite(v[1])}
    lab = select.pick(ok, key=lambda k: (select.order_key(pool[k][0]), k[0]))
    return pool[lab][0], lab[0], pool[lab][1]


def hp_json(hp) -> dict:
    d = {a: float(getattr(hp, a, 0.0)) for a in space.AXES}   # полная точность (JSON хранит repr float без потерь)
    d["max_trees"] = int(getattr(hp, "max_trees", space.MAX_TREES))
    return d


def run_study(p, name: str, cache: Path, n: int = space.N_TPE) -> list:
    """22 trials последовательно; результаты фолдов кэшируются (resume): [{params, scores, J}]."""
    done = json.loads(cache.read_text()) if cache.exists() else []
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=space.STUDY_SEED, n_startup_trials=space.N_STARTUP))
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def obj(trial):
        hp = space.suggest(trial)
        k = trial.number
        if k < len(done):
            return done[k]["J"]
        sc = engine_c.scores(p, name, {Y: hp for Y in space.ALL_YEARS})
        done.append({"k": k, "params": hp_json(hp), "scores": {str(Y): v for Y, v in sc.items()}, "J": J(sc)})
        cache.write_text(json.dumps(done))
        return done[k]["J"]

    study.optimize(obj, n_trials=n, n_jobs=1)
    return done


def hp_of(params: dict) -> fitx.HPX:
    return fitx.HPX(int(params["depth"]), params["min_child"], params["l2"], params["lr"], params["subsample"], params["colsample"],
                    int(params.get("max_trees", space.MAX_TREES)), params["gamma"], params["alpha"])
