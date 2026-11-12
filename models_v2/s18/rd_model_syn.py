"""Синтетика S-RCD-IMPL-1 (M.G4): реальные признаки C-D (пары D_M, π, β, ln B), RV = F*·Z, Z — логнормальная σ = 0,5 плюс 1 % скачков ×30 (E Z = 1). Форма CD. Только расчёт."""
from __future__ import annotations

import copy

import numpy as np
import torch

from models_v2.s18 import r_loss as RL
from models_v2.s18 import rd_model as RM
from models_v2.s18.rd_model_chk import load

REPS, SPLIT = 20, 1100


def make(D, lv, seed: int, shift=None, slope_zc: float = 0.0):
    """RV = F*·Z; ln C* = ln C(0) + shift + slope_zc·zc (при θ = (shift, 0, 0, slope_zc, 0, 0))."""
    rng, n = np.random.default_rng(seed), len(lv.pi)
    th = np.zeros(6)
    th[0], th[3] = 0.0 if shift is None else shift, slope_zc
    V = RL.view(D, lv, np.arange(n))
    F = torch.exp(RL.ln_f(RM.lnc_of(torch.as_tensor(th), V, "CD"), V)).numpy()
    Z = np.exp(0.5 * rng.normal(size=n) - 0.125) * np.where(rng.random(n) < 0.01, 30.0, 1.0) / (1.0 + 0.29)
    Dc = copy.copy(D)
    Dc.rv = F * Z
    return Dc


def stats_on(D, lv, sel) -> tuple:
    V = RL.view(D, lv, sel)
    x0 = (V.rv * torch.exp(-RL.ln_f(RM.lnc_of(torch.zeros(6, dtype=torch.float64), V, "CD"), V))).numpy()
    return RL.clip_stats(x0, np.arange(len(sel)))


def fit_on(D, lv, sel, lam: float, mode: str, cm=None) -> np.ndarray:
    c, m = stats_on(D, lv, sel) if cm is None else cm
    return RM.fit(RL.view(D, lv, sel), "CD", lam, c, m, mode)[0]


def qlike_on(D, lv, sel, th) -> float:
    V = RL.view(D, lv, sel)
    return float(RL.ell(V.rv.numpy(), RL.ln_f(RM.lnc_of(torch.as_tensor(th), V, "CD"), V).numpy()).mean())


def synth() -> list:
    bad, (D, x, lv) = [], load()
    n = len(lv.pi)
    tr, te = np.arange(SPLIT), np.arange(SPLIT, n)
    for lam in RL.GRID:
        r, nz = [], []
        for s in range(REPS):
            Dc = make(D, lv, s)
            th = fit_on(Dc, lv, tr, lam, "R")
            r.append(qlike_on(Dc, lv, te, th) / qlike_on(Dc, lv, te, np.zeros(6)) - 1.0)
            nz.append(float(np.abs(th).max() > 1e-6))
        print("(а) нулевой сигнал, λ = %g: QLIKE вне выборки R − θ=0 = %+.4f %%, доля θ ≠ 0 = %.2f" % (lam, 100 * np.mean(r), np.mean(nz)))
        if lam == RL.GRID[0] and np.mean(r) > 1e-3:
            bad.append("(а) λ = 0,1: %+.4f %%" % (100 * np.mean(r)))
    for mode in RM.MODES:
        al = np.arange(n)
        est = np.array([fit_on(make(D, lv, 100 + s, shift=-0.3), lv, al, 1e-3, mode) for s in range(REPS)])
        med = float(np.median(est[:, 0]))
        print("(б) сдвиг уровня −0,3 (режим %s, c, m по сдвинутым данным, как в проходе): медиана θ̂0 = %+.4f" % (mode, med))
        if mode == "R":    # ↺ m калибрует уровень: при θ = 0 балл уровня 0 на любых данных (M.G1), равномерный сдвиг R не видит; критерий (б) для R — при c, m по несдвинутым данным
            est = np.array([fit_on(make(D, lv, 100 + s, shift=-0.3), lv, al, 1e-3, mode, stats_on(make(D, lv, 100 + s), lv, al)) for s in range(REPS)])
            med = float(np.median(est[:, 0]))
            print("(б) режим R, c, m по несдвинутым данным (↺ отступление от формулировки спецификации): медиана θ̂0 = %+.4f" % med)
        bad += ["(б) %s: θ̂0 = %+.4f" % (mode, med)] * (not (med < 0 and abs(med + 0.3) <= 0.1))
        est = np.array([fit_on(make(D, lv, 200 + s, slope_zc=0.3), lv, np.arange(n), 1e-3, mode) for s in range(REPS)])
        med = float(np.median(est[:, 3]))
        print("(в) наклон 0,3 по zc (режим %s): медиана θ̂_c1 = %+.4f" % (mode, med))
        bad += ["(в) %s: θ̂_c1 = %+.4f" % (mode, med)] * (abs(med - 0.3) > 0.1)
    return bad
