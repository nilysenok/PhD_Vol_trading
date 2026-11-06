"""Ж-214, V5 Q1/Q2: HAR-J в УРОВНЯХ с QLIKE-подгонкой и ограничениями β >= 0.
  σ̂²(d) = β0 + β_d RV_d + β_w RV_w + β_m RV_m + β_J J_d  (Q1);  β_J ≡ 0 (Q2).  Положительность — из ограничений, не из exp.
QLIKE (среднее ln f + y/f, y = RV_{t+1}) на закрытых парах (t+1 <= d), расширяющееся окно, ежедневная переоценка, состояние между днями не переносится.
Проекция: границы L-BFGS-B на колонках, отмасштабированных на средние обучающей выборки (масштаб знак β не меняет); нижняя граница свободного члена 1e-8.
Два холодных старта (равные доли среднего y; OLS в уровнях с обрезкой), берётся лучший по значению цели. `nonneg=False` — МУТАНТ «без ограничений»: ловится тестом."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from models_v2.s9 import hv2_world as W

FLOOR = 1e-8


def levels(w: W.World, with_j: bool = True) -> np.ndarray:
    """n x k: [1, RV_d, RV_w, RV_m, (J)] в %^2 (строка t)."""
    p = w.p
    cols = [np.ones(len(p.rv)), p.rv, np.exp(p.X[:, 1]), np.exp(p.X[:, 2])]
    if with_j:
        cols.append(np.expm1(p.X[:, 3]))
    return np.column_stack(cols)


def _fg(th, S, y):
    f = np.einsum("ij,j->i", S, th)
    if not (f > 0).all():
        return 1e10, np.zeros_like(th)
    return float(np.mean(np.log(f) + y / f)), np.einsum("ij,i->j", S, 1.0 / f - y / (f * f)) / len(y)


def fit_levels(Z: np.ndarray, y: np.ndarray, nonneg: bool = True) -> np.ndarray:
    """β (в исходном масштабе колонок Z) = argmin среднего QLIKE при β >= 0 (β0 >= 1e-8)."""
    if len(y) <= Z.shape[1] or not (np.asarray(y) > 0).all() or not np.isfinite(Z).all():
        raise ValueError("QLIKE в уровнях: пусто/мало пар, RV <= 0 или нефинитные признаки (n=%d, k=%d)" % (len(y), Z.shape[1]))
    c = Z.mean(axis=0)
    dead = c == 0                        # колонка из нулей (например, J ≡ 0 в окне): β = 0 по определению
    c[dead] = 1.0
    S, k, yb = Z / c, Z.shape[1], float(np.mean(y))
    th_a = np.r_[0.1 * yb, np.full(k - 1, 0.9 * yb / (k - 1))]
    th_b = np.maximum(np.linalg.lstsq(S, y, rcond=None)[0], 0.0)
    th_b[0] = max(th_b[0], FLOOR)
    th_a[dead] = th_b[dead] = 0.0
    bnd = [(FLOOR, None)] + [(0.0, None)] * (k - 1) if nonneg else [(None, None)] * k
    best = None
    for th0 in (th_a, th_b if nonneg else np.linalg.lstsq(S, y, rcond=None)[0]):
        r = minimize(_fg, th0, args=(S, y), jac=True, method="L-BFGS-B", bounds=bnd,
                     options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 2000, "maxfun": 10000})
        if best is None or r.fun < best.fun:
            best = r
    return best.x / c


def run(w: W.World, with_j: bool = True, nonneg: bool = True) -> tuple:
    """-> (σ̂² на сетке w.el, β n x k). Прогноз на d = строка i: пары t+1 <= i."""
    Z = levels(w, with_j)
    s2, B = np.empty(len(w.el)), np.empty((len(w.el), Z.shape[1]))
    for a, i in enumerate(w.el):
        idx = W.pairs_idx(w.p.seg, Z, int(i))
        B[a] = fit_levels(Z[idx], w.p.rv[idx + 1], nonneg)
        s2[a] = float(np.einsum("j,j->", B[a], Z[i]))
    return s2, B
