"""D: ежедневные прогнозы бустера (XGB/LGBM) с окном и весами. Пул spawn <= 6. Прогноз на d: пары с t+1 <= d, HP — по году даты d (rows.year_of)."""
from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np
import pandas as pd

from models_v2.s4 import fit, rows, runner
from models_v2.s5.c import engine_c, fitx
from models_v2.s5.d import lgbfit

MAX_PROCS = 6
W1008, HL504 = 1008, 504
_W: engine_c.W | None = None


def _init(w):
    global _W
    _W = w


def day_fit(F, dz, d: int, hp, seed: int, lib: str = "xgb", win: int = 0, hl: float = 0.0) -> dict:
    """Обучение на данных <= d; win — последние win пар (0 — все), hl — полураспад весов в днях (0 — без весов)."""
    idx = runner.pairs_upto(dz, F, d)
    if win:
        idx = idx[-win:]
    w = fit.weights(np.arange(len(F.rv)), d, hl) if hl else None
    f = fitx.fit_i1x if lib == "xgb" else lgbfit.fit_lgbx
    return f(dz.X, dz.y, idx, hp, w, None, seed)


def _day_job(a):
    name, d, hp, seed, lib, win, hl = a
    t0 = time.time()
    m = day_fit(_W.F, _W.dzs[name], d, hp, seed, lib, win, hl)
    return d, {"f": float(m["predict"](_W.dzs[name].X[[d]])[0]), "k": m["k"], "s2_es": m["s2_es"], "s2_tr": m["s2_tr_prelim"], "secs": time.time() - t0}


def pool(w: engine_c.W, procs: int = MAX_PROCS):
    return mp.get_context("spawn").Pool(min(procs, MAX_PROCS), initializer=_init, initargs=(w,))


def daily(p, w: engine_c.W, name: str, hp_y: dict, seed: int = fit.SEED, lib: str = "xgb", win: int = 0, hl: float = 0.0) -> tuple:
    """-> (DataFrame d, target_date, sigma2, rv; статистика). Формат как у engine_c.daily."""
    F, dz = w.F, w.dzs[name]
    res = dict(p.imap_unordered(_day_job, [(name, int(d), hp_y[rows.year_of(F, int(d))], seed, lib, win, hl) for d in w.grid], chunksize=8))
    i = np.asarray(w.grid, int)
    f = np.array([res[int(d)]["f"] for d in i])
    s2 = np.array([res[int(d)]["s2_es"] for d in i])
    fc = pd.DataFrame({"d": F.dates[i], "target_date": F.dates[i + 1], "sigma2": np.exp(dz.off[i] + f + s2 / 2), "rv": F.rv[i + 1]})
    st = {"trees": float(np.mean([res[int(d)]["k"] for d in i])), "mse_train": float(np.mean([res[int(d)]["s2_tr"] for d in i])),
          "mse_block": float(s2.mean()), "n_fits": len(i), "secs": float(sum(res[int(d)]["secs"] for d in i)), "n_feat": int(dz.X.shape[1])}
    return fc, st
