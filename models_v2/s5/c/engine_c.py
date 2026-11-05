"""C: фолды LOYO и ежедневные прогнозы бустера с годовыми HP (HPX или fit.HP) и seed. Пул spawn <= 4. Кэш прогнозов — по ключу."""
from __future__ import annotations

import hashlib
import multiprocessing as mp
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s4 import fit, rows, runner, select
from models_v2.s5.c import fitx, space

MAX_PROCS = 4


@dataclass
class W:
    F: object
    grid: np.ndarray
    dzs: dict       # {имя: Dz}


_W: W | None = None


def _init(w):
    global _W
    _W = w


def fold(F, dz, hp, Tc: int, y: int, seed: int = fit.SEED) -> float:
    """QLIKE(б) на целях года y; обучение — пары с t+1 <= Tc без года y (как loyo.fold / runner.fold_score)."""
    tr, val = runner.fold_split(F, dz, Tc, y)
    if len(val) < 50 or len(tr) < 2 * fit.MIN_ES + fit.EMBARGO:
        return float("nan")
    m = fitx.fit_i1x(dz.X, dz.y, tr, hp, None, None, seed)
    s2 = np.exp(dz.off[val] + m["predict"](dz.X[val]) + m["s2_es"] / 2)
    return float(metrics.qlike(F.rv[val + 1], s2).mean())


def _fold_job(a):
    key, name, hp, Y, y, seed = a
    return key, fold(_W.F, _W.dzs[name], hp, select.cut_row(_W.F, Y), y, seed)


def _day_job(a):
    name, d, hp, seed = a
    t0 = time.time()
    dz = _W.dzs[name]
    idx = runner.pairs_upto(dz, _W.F, d)
    m = fitx.fit_i1x(dz.X, dz.y, idx, hp, None, None, seed)
    return d, {"f": float(m["predict"](dz.X[[d]])[0]), "k": m["k"], "s2_es": m["s2_es"], "s2_tr": m["s2_tr_prelim"], "secs": time.time() - t0}


def pool(w: W, procs: int = MAX_PROCS):
    return mp.get_context("spawn").Pool(min(procs, MAX_PROCS), initializer=_init, initargs=(w,))


def scores(p, name: str, hp_y: dict, seed: int = fit.SEED, years=space.ALL_YEARS) -> dict:
    """{Y: score_Y}: среднее по y в [2015, Y-1] фолдов; hp_y[Y] — HP года Y (цели < 01.01.Y)."""
    jobs = [((Y, y), name, hp_y[Y], Y, y, seed) for Y in years for y in range(select.FIRST_Y, Y)]
    res = {}
    for (Y, y), q in p.imap_unordered(_fold_job, jobs, chunksize=4):
        res.setdefault(Y, []).append(q)
    return {Y: float(np.nanmean(res[Y])) for Y in years}


def crit(sc: dict) -> float:
    return float(np.mean(list(sc.values())))


def daily(p, w: W, name: str, hp_y: dict, seed: int = fit.SEED) -> tuple:
    """-> (DataFrame d, target_date, sigma2, rv; статистика). HP — по году даты d (rows.year_of)."""
    F, dz = w.F, w.dzs[name]
    tasks = [(name, int(d), hp_y[rows.year_of(F, int(d))], seed) for d in w.grid]
    res = dict(p.imap_unordered(_day_job, tasks, chunksize=8))
    i = np.asarray(w.grid, int)
    f = np.array([res[int(d)]["f"] for d in i])
    s2 = np.array([res[int(d)]["s2_es"] for d in i])
    fc = pd.DataFrame({"d": F.dates[i], "target_date": F.dates[i + 1], "sigma2": np.exp(dz.off[i] + f + s2 / 2), "rv": F.rv[i + 1]})
    st = {"trees": float(np.mean([res[int(d)]["k"] for d in i])), "mse_train": float(np.mean([res[int(d)]["s2_tr"] for d in i])),
          "mse_block": float(s2.mean()), "n_fits": len(i), "secs": float(sum(res[int(d)]["secs"] for d in i)), "n_feat": int(dz.X.shape[1])}
    return fc, st


def hp_key(hp_y: dict, seed: int) -> str:
    s = repr(sorted((Y, space.hp_dict(h)) for Y, h in hp_y.items())) + str(seed)
    return hashlib.sha256(s.encode()).hexdigest()[:12]
