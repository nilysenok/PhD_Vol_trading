"""Спецификации строк, фолды LOYO и ежедневные прогнозы LightGBM (поток LGBM, волна B). Пул spawn <= 4 процессов."""
from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s4 import fit, rows, runner, select
from models_v2.s5.b1 import designs
from models_v2.s6.lgbm import fitl
from models_v2.s6.lgbm.grid import Hp

MAX_PROCS = 4
YEARS = tuple(range(2017, 2027))
FOLDS = tuple((Y, y) for Y in YEARS for y in range(select.FIRST_Y, Y))   # 65 фолдов
_S: dict = {}


@dataclass(frozen=True)
class Spec:
    name: str
    kind: str                 # ln | lvl | gamma | qlike
    X: np.ndarray
    y: np.ndarray             # y[t] — цель пары (x_t, y_t): ln-цель ('ln') или RV/S
    off: np.ndarray           # ln S (σ̂² = exp(off + …))
    wt: tuple = ()            # () | ('rvw', γ) | ('decay', полураспад)


def base_spec(L: dict, name: str, kind: str, wt: tuple = ()) -> Spec:
    dz = L["dz0"]
    y = dz.y if kind == "ln" else np.exp(dz.y)
    return Spec(name, kind, dz.X, y, dz.off, wt)


def with_X(sp: Spec, X: np.ndarray, name: str) -> Spec:
    return replace(sp, name=name, X=X)


def weights(F, sp: Spec, idx: np.ndarray, T: int):
    """Веса по строкам t (или None). rvw: w ∝ RV_w^(−γ), нормировка по парам окна (b1.designs); decay: 0,5^((T − t)/hl) (fit.weights)."""
    if not sp.wt:
        return None
    if sp.wt[0] == "rvw":
        return designs.weights_l5(F, float(sp.wt[1]), idx)
    return fit.weights(np.arange(len(F.rv)), T, float(sp.wt[1]))


def pairs(F, sp: Spec, T: int) -> np.ndarray:
    return fit.pair_index(F.seg, sp.X, sp.y, (0, 1, 2, 3), T)


def fold_q(F, sp: Spec, eps, hp: Hp, Y: int, y: int) -> float:
    """QLIKE (б) на целях года y модели, обученной на парах с целями < 01.01.Y кроме года y (HP года Y)."""
    T = select.cut_row(F, Y)
    tr, val = runner.fold_split(F, sp, T, y)
    if len(val) < 50 or len(tr) < 2 * fit.MIN_ES + fit.EMBARGO:
        return float("nan")
    m = fitl.fit_l(sp.X, sp.y, tr, hp, sp.kind, weights(F, sp, tr, T))
    s2, _ = fitl.sigma2(sp.kind, val, m["predict"](sp.X[val]), m["s2_es"], sp.off, np.exp(sp.off), eps)
    return float(metrics.qlike(F.rv[val + 1], s2).mean())


def day_fc(F, sp: Spec, hp: Hp, d: int) -> dict:
    t0 = time.time()
    idx = pairs(F, sp, d)
    m = fitl.fit_l(sp.X, sp.y, idx, hp, sp.kind, weights(F, sp, idx, d))
    return {"f": float(m["predict"](sp.X[[d]])[0]), "s2": m["s2_es"], "k": m["k"], "k_cap": m["k_cap"], "nf": m["n_nonfinite_es"], "secs": time.time() - t0}


def _init(F, specs, eps):
    _S.update(F=F, specs=specs, eps=eps)


def _fold_job(a):
    name, hp, Y, y = a
    return (name, Y, y), fold_q(_S["F"], _S["specs"][name], _S["eps"], hp, Y, y)


def _day_job(a):
    name, hp, d = a
    return (name, d), day_fc(_S["F"], _S["specs"][name], hp, d)


def pool(F, specs: dict, eps, procs: int = MAX_PROCS):
    return mp.get_context("spawn").Pool(min(procs, MAX_PROCS), initializer=_init, initargs=(F, specs, eps))


def run_folds(p, name: str, hp_y: dict) -> dict:
    """hp_y: {Y: Hp} -> {(Y, y): QLIKE}."""
    return dict(p.imap_unordered(_fold_job, [(name, hp_y[Y], Y, y) for Y, y in FOLDS], chunksize=2))


def run_days(p, F, name: str, hp_y: dict, grid) -> dict:
    return dict(p.imap_unordered(_day_job, [(name, hp_y[rows.year_of(F, int(d))], int(d)) for d in grid], chunksize=8))


def assemble(F, sp: Spec, eps, grid, res: dict, name: str) -> tuple:
    """-> (DataFrame d, target_date, sigma2, rv; статистика устойчивости). σ̂² обязана быть конечной и > 0."""
    i = np.asarray(grid, int)
    r = [res[(name, int(d))] for d in i]
    f, s2 = np.array([x["f"] for x in r]), np.array([x["s2"] for x in r])
    sig, n_eps = fitl.sigma2(sp.kind, i, f, s2, sp.off, np.exp(sp.off), eps)
    bad = int((~(np.isfinite(sig) & (sig > 0))).sum())
    if bad:
        raise ValueError("%s: %d неположительных/нечисловых σ̂²" % (name, bad))
    fc = pd.DataFrame({"d": F.dates[i], "target_date": F.dates[i + 1], "sigma2": sig, "rv": F.rv[i + 1]})
    k = np.array([x["k"] for x in r])
    st = {"trees": float(k.mean()), "k_min": int(k.min()), "share_k1": float((k <= 1).mean()), "share_k_cap": float(np.mean([x["k_cap"] for x in r])),
          "n_nonfinite_es": int(sum(x["nf"] for x in r)), "n_eps_floor": n_eps, "n_fits": len(i), "secs": float(sum(x["secs"] for x in r)),
          "n_feat": int(sp.X.shape[1]), "mse_block": float(np.nanmean(s2)) if np.isfinite(s2).any() else float("nan")}
    return fc, st
