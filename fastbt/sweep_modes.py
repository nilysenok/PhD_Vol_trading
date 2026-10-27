"""Режимы перебора (Ж-031, Э1-а): OAT, факторный, Моррис (своя реализация), Optuna TPE.

SALib в окружении нет (решение класса B, tolkovaniya.md, 27.10 18:30): элементарные эффекты считаются здесь.
Функции не знают про движок: получают `grids` {параметр: [значения]} и вызываемый `f(params) -> float`.
"""
from __future__ import annotations

from itertools import product

import numpy as np

MAX_JOBS = 8
MORRIS_R = 10


def oat_points(base: dict, grids: dict) -> list:
    """По одному параметру, остальные на исходных; базовая точка — один раз, первой."""
    pts, seen = [dict(base)], {tuple(sorted(base.items()))}
    for name, vals in grids.items():
        for v in vals:
            p = {**base, name: v}
            k = tuple(sorted(p.items()))
            if k not in seen:
                seen.add(k)
                pts.append(p)
    return pts


def factorial_points(base: dict, grids: dict, names=None, points=None) -> list:
    """Факторный: полный перебор сеток `names` (остальные на исходных) либо явный список `points`."""
    if points is not None:
        return [{**base, **p} for p in points]
    names = list(names or [])
    if not names:
        raise ValueError("факторный режим: нужен список параметров (names) или точек (points)")
    return [{**base, **dict(zip(names, c))} for c in product(*(grids[n] for n in names))]


def morris_trajectory(levels: dict, rng: np.random.Generator) -> list:
    """Одна траектория: старт на случайных уровнях сетки, затем k шагов в случайном порядке параметров,
    каждый — на ±1 уровень (вверх, если можно, иначе вниз; направление случайно). [(индексы, имя, du)]."""
    names = list(levels)
    L = {n: len(levels[n]) for n in names}
    cur = {n: int(rng.integers(0, L[n])) for n in names}
    out = [(dict(cur), None, 0.0)]
    for n in rng.permutation(names):
        n = str(n)
        opts = [d for d in (1, -1) if 0 <= cur[n] + d < L[n]]
        d = int(rng.choice(opts))
        cur[n] += d
        out.append((dict(cur), n, d / (L[n] - 1)))
    return out


def morris_effects(f, levels: dict, r: int = MORRIS_R, seed: int = 0, base: dict | None = None) -> dict:
    """Элементарные эффекты EE = Δf/Δu (u = номер уровня/(L−1) ∈ [0,1]), r траекторий.
    Возвращает {имя: dict(mu, mu_star, sigma, n)}: μ — среднее, μ* — среднее модулей, σ — std (ddof=1)."""
    if any(len(v) < 2 for v in levels.values()):
        raise ValueError("Моррис: в сетке каждого параметра нужно ≥ 2 уровня")
    rng = np.random.default_rng(seed)
    base = base or {}
    ee = {n: [] for n in levels}
    for _ in range(r):
        prev = None
        for idx, name, du in morris_trajectory(levels, rng):
            val = f({**base, **{n: levels[n][i] for n, i in idx.items()}})
            if name is not None:
                ee[name].append((val - prev) / du)
            prev = val
    out = {}
    for n, v in ee.items():
        a = np.asarray(v, float)
        out[n] = dict(mu=float(a.mean()), mu_star=float(np.abs(a).mean()),
                      sigma=float(a.std(ddof=1)) if len(a) > 1 else 0.0, n=len(a))
    return out


def optuna_run(sweep, n_trials: int, seed: int = 0, lam: float = 0.5, n_jobs: int = 1) -> list:
    """Optuna TPE: цель — среднее годовых Шарпов − λ·σ по годам train; pruning (MedianPruner) по первому году.
    `sweep.partial(params, years)` возвращает Шарп первого года, `sweep.objective(params, lam)` — цель.
    Воспроизводимость при фиксированном seed гарантирована при n_jobs = 1 (при n_jobs > 1 порядок проб гонка потоков)."""
    import optuna
    if not 1 <= n_jobs <= MAX_JOBS:
        raise ValueError(f"n_jobs должно быть в 1..{MAX_JOBS}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0))
    grids = sweep.spec.grids

    def obj(trial):
        p = {**sweep.spec.base, **{n: trial.suggest_categorical(n, list(v)) for n, v in grids.items()}}
        first = sweep.partial(p)
        trial.report(first, 0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return sweep.objective(p, lam)

    study.optimize(obj, n_trials=n_trials, n_jobs=n_jobs)
    return [dict(number=t.number, state=t.state.name, value=t.value, params=dict(t.params)) for t in study.trials]
