"""V4-B: иерархическая оценка параметров операторов на закрытых парах. θ_i = θ_panel + δ_i, штраф κ·Σδ² (κ = 250 пар-эквивалентов), приор w₀ = 0,5 (B6: 1/3 каждый; b₀ = 0; p₀ = 0) силой 250 пар;
до первых 60 закрытых пар — приор. Гиперпараметры (KAPPA, LAM, MIN_PAIRS) фиксированы объявлением и не выбираются по результатам (G9/G10)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from models_v2.s11 import v4_b_ops as OP, v4_b_solve as SV

KAPPA = 250.0
LAM = 250.0
MIN_PAIRS = 60
TH0 = np.full(3, 1.0 / 3.0)


@dataclass
class Train:
    rv: np.ndarray
    H: np.ndarray
    M: np.ndarray
    lH: np.ndarray
    lM: np.ndarray
    lK2: np.ndarray
    lK3: np.ndarray
    tid: np.ndarray       # компактный номер бумаги 0..n-1 (только бумаги с закрытыми парами)
    n: int


def make_train(rv, H, K2, K3, tid_full) -> tuple:
    """-> (Train, present): present — исходные номера бумаг, имеющих хотя бы одну закрытую пару."""
    present, tid = np.unique(np.asarray(tid_full), return_inverse=True)
    H, K2, K3 = (np.asarray(x, float) for x in (H, K2, K3))
    M = OP.mean_m(K2, K3)
    return Train(np.asarray(rv, float), H, M, np.log(H), np.log(M), np.log(K2), np.log(K3), tid, len(present)), present


def curvature(T: Train) -> float:
    return float(np.mean((T.lM - T.lH) ** 2))


def _layout(cfg: str, n: int) -> tuple:
    """(x0, lo, hi) нейтральной стартовой точки: [w_p, w_1..w_n, (p) | (b_p, b_1..b_n)]."""
    x0, lo, hi = [OP.W0] * (n + 1), [0.0] * (n + 1), [1.0] * (n + 1)
    if cfg == "B4":
        x0, lo, hi = x0 + [0.0], lo + [-OP.P_BOUND], hi + [OP.P_BOUND]
    if cfg == "B5":
        x0, lo, hi = x0 + [OP.B0] * (n + 1), lo + [-OP.B_BOUND] * (n + 1), hi + [OP.B_BOUND] * (n + 1)
    return np.array(x0), np.array(lo), np.array(hi)


def _unpack(cfg, x, n):
    p, bp, b = 0.0, 0.0, np.zeros(n)
    if cfg == "B4":
        p = float(x[n + 1])
    if cfg == "B5":
        bp, b = float(x[n + 1]), x[n + 2:]
    return float(x[0]), x[1:n + 1], p, bp, b


def _obj(cfg, T, s2, kappa, lam):
    n, N = T.n, len(T.rv)
    c, l = kappa * s2, lam * s2

    def f(x):
        wp, w, p, bp, b = _unpack(cfg, x, n)
        F, dw, dp, db = OP.forward(cfg, T.H, T.M, w[T.tid], p, b[T.tid])
        gF = 1.0 / F - T.rv / F ** 2
        dev, db_ = w - wp, b - bp
        val = OP.qloss(T.rv, F).sum() + 0.5 * c * float(dev @ dev) + 0.5 * l * (wp - OP.W0) ** 2
        g = np.zeros_like(x)
        g[1:n + 1] = np.bincount(T.tid, gF * dw, minlength=n) + c * dev
        g[0] = -c * dev.sum() + l * (wp - OP.W0)
        if cfg == "B4":
            g[n + 1] = float((gF * dp).sum())
        if cfg == "B5":
            val += 0.5 * kappa * float(db_ @ db_) + 0.5 * lam * (bp - OP.B0) ** 2
            g[n + 2:] = np.bincount(T.tid, gF * db, minlength=n) + kappa * db_
            g[n + 1] = -kappa * db_.sum() + lam * (bp - OP.B0)
        return val / N, g / N
    return f


def _pack(cfg, x, n, ok) -> dict:
    wp, w, p, bp, b = _unpack(cfg, x, n)
    return {"w_p": wp, "w": np.array(w), "p": p, "b_p": bp, "b": np.array(b), "ok": bool(ok)}


def _lbfgs(cfg, T, s2, kappa, lam, starts) -> tuple:
    _, lo, hi = _layout(cfg, T.n)
    best = None
    for x0 in starts:
        r = minimize(_obj(cfg, T, s2, kappa, lam), np.clip(x0, lo, hi), jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)), options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-10})
        g = r.jac
        pg = np.where(((r.x <= lo + 1e-12) & (g > 0)) | ((r.x >= hi - 1e-12) & (g < 0)), 0.0, g)
        if best is None or r.fun < best[1]:
            best = (r.x, float(r.fun), bool(np.abs(pg).max() < 1e-6))
    return best


def fit(cfg: str, T: Train, prev=None, kappa: float | None = None, lam: float | None = None) -> dict:
    """Оценка на закрытых парах T. prev — решение прошлого месяца (тёплый старт, B4 — вторая нейтральная). Ключи: w_p, w (n,), p, b_p, b (n,), theta_p (3,), theta (n, 3), ok."""
    kappa, lam = KAPPA if kappa is None else kappa, LAM if lam is None else lam
    n, s2 = T.n, curvature(T)
    if cfg == "B6":
        x0 = np.tile(TH0, (n + 1, 1)) if prev is None or prev.shape[0] != n + 1 else prev
        X, _, ok = SV.solve_b6(T, s2, kappa, lam, TH0, x0)
        return {"theta_p": X[0], "theta": X[1:], "ok": ok, "_x": X}
    x0, lo, hi = _layout(cfg, n)
    if cfg == "B2":
        x, _, ok = SV.newton_box(SV.b2_fgh(T, s2, kappa, lam, OP.W0), x0, lo, hi)
    else:
        starts = [x0] + ([prev] if prev is not None and len(prev) == len(x0) else [])
        x, _, ok = _lbfgs(cfg, T, s2, kappa, lam, starts)
    out = _pack(cfg, x, n, ok)
    out["_x"] = x
    return out


def prior_params(cfg: str) -> dict:
    """Параметры до первых MIN_PAIRS закрытых пар: приор (w = 0,5; p = 0; b = 0; B6 — 1/3 каждому)."""
    return {"w_p": OP.W0, "p": 0.0, "b_p": OP.B0, "theta_p": TH0.copy()}
