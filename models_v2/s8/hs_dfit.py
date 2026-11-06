"""Ж-212 (HS7, G10): Component Hybrid. Компоненты φ = (ln RV_d, ln RV_w, ln RV_m, ln(1+J), ln(RS−/RS+)). D0: ln σ̂² = b0 + Σ b_k φ_k, коэффициенты — минимум QLIKE по закрытым парам
(t+1 <= d, один сегмент), ежедневная переподгонка. Dn_k: линейный член b_kφ_k заменён формой s_k(φ_k) — бустинг пней (глубина 1, 100 деревьев, lr 0,05, объектив QLIKE-gamma),
offset = линейная часть D0-full без k + c0; Dx_k: то же с X_t и глубиной 2. σ̂² = exp(·) > 0 по построению (отрицательных прогнозов нет)."""
from __future__ import annotations

import multiprocessing as mp

import numpy as np
import xgboost as xgb

from models_v2.s4 import fit
from models_v2.s8 import hs_qfit

PHI = ("d", "w", "m", "J", "a")
SHAPE_Q = np.linspace(0.02, 0.98, 25)
PROCS = 4
_S: dict = {}


def phi_matrix(F) -> np.ndarray:
    x = F.X
    return np.column_stack([x[:, 0], x[:, 1], x[:, 2], x[:, 3], np.log((x[:, 8] + 1e-12) / (x[:, 7] + 1e-12))])


def target(F, eps) -> np.ndarray:
    rvn = np.append(F.rv[1:], np.nan)
    return np.maximum(rvn, eps)


def lin_fit(Phi, y, idx, cols, gam0=None) -> np.ndarray:
    """QLIKE-подгонка (b0, b_cols) по парам idx."""
    X = np.column_stack([np.ones(len(idx)), Phi[idx][:, list(cols)]])
    return hs_qfit.fit(X, y[idx], np.zeros(len(idx)), gam0=gam0)


def lin_run(F, Phi, y, grid, cols) -> np.ndarray:
    """σ̂² на сетке линейной модели D0 с компонентами cols; ежедневная переподгонка с тёплым стартом."""
    out, gam = np.empty(len(grid)), None
    for i, d in enumerate(grid):
        idx = fit.pair_index(F.seg, Phi, y, tuple(range(Phi.shape[1])), int(d))
        gam = lin_fit(Phi, y, idx, cols, gam)
        out[i] = np.exp(gam[0] + (gam[1:] * Phi[d, list(cols)]).sum())
    return out


def stump_fit(Xs, y, off, depth: int, seed: int = 42):
    p = {"objective": "reg:gamma", "max_depth": depth, "eta": 0.05, "tree_method": "hist", "nthread": 1, "seed": seed, "base_score": 0.0, "disable_default_eval_metric": 1}
    b = xgb.train(p, xgb.DMatrix(Xs, label=y, base_margin=off), 100)
    return lambda Z: b.predict(xgb.DMatrix(np.atleast_2d(Z), base_margin=np.zeros(len(np.atleast_2d(Z)))), output_margin=True).astype(float)


def nl_day(Phi, y, seg, Xs, k: int, depth: int, d: int, qgrid) -> tuple:
    """Dn/Dx на дату d: -> (ln σ̂², форма s_k на qgrid (только пни по φ_k; для Dx — NaN))."""
    idx = fit.pair_index(seg, Phi, y, tuple(range(Phi.shape[1])), d)
    gam = lin_fit(Phi, y, idx, range(Phi.shape[1]))
    rest = [j for j in range(Phi.shape[1]) if j != k]
    lin_rest = lambda r: gam[0] + (gam[1:][rest] * Phi[r][:, rest]).sum(axis=1)
    off = lin_rest(idx)
    c0 = float(np.log(np.mean(y[idx] / np.exp(off))))
    pr = stump_fit(Xs[idx], y[idx], off + c0, depth)
    ln_s2 = float(lin_rest(np.array([d]))[0] + c0 + pr(Xs[[d]])[0])
    shape = pr(np.column_stack([qgrid, np.zeros((len(qgrid), Xs.shape[1] - 1))]))if Xs.shape[1] == 1 else np.full(len(qgrid), np.nan)
    return ln_s2, shape


def init(Phi, y, seg, Xs, k, depth, qgrid):
    _S.update(Phi=Phi, y=y, seg=seg, Xs=Xs, k=k, depth=depth, qgrid=qgrid)


def day_job(d):
    S = _S
    return d, nl_day(S["Phi"], S["y"], S["seg"], S["Xs"], S["k"], S["depth"], d, S["qgrid"])


def nl_run(F, Phi, y, grid, Xs, k: int, depth: int, procs: int = PROCS) -> tuple:
    """-> (σ̂² на сетке, форма s_k по датам (n×len(SHAPE_Q)), узлы φ_k (квантили по строкам до сетки — только прошлое не требуется: узлы для отображения))."""
    qgrid = np.nanquantile(Phi[:int(grid[0]), k], SHAPE_Q)
    with mp.get_context("spawn").Pool(min(procs, PROCS), initializer=init, initargs=(Phi, y, F.seg, Xs, k, depth, qgrid)) as p:
        res = dict(p.imap_unordered(day_job, [int(d) for d in grid], chunksize=8))
    s2 = np.exp([res[int(d)][0] for d in grid])
    return s2, np.array([res[int(d)][1] for d in grid]), qgrid
