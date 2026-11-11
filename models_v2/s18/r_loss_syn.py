"""Синтетика S-R-IMPL-1 (L.G4): `cm_model.forward`, n = 2000, Z — логнормальная σ = 0,5 плюс 1 % скачков ×30 (нормировано E Z = 1). Только расчёт."""
from __future__ import annotations

import numpy as np
import torch

from models_v2.s18 import cm_model as CM
from models_v2.s18 import r_loss as RL


def synth_data(n: int, seed: int, ths, pi: float = 0.5, sig: float = 0.5, jump: float = 0.01):
    rng = np.random.default_rng(seed)
    c8, sc, sp = np.exp(0.3 * rng.normal(size=n)), np.full(n, 0.4), np.full(n, 0.5)
    kw = dict(c8=c8, sc=sc, sp=sp, su=np.full(n, 0.1), Hc=sc * c8 * np.exp(0.25 * rng.normal(size=n)), Hp=sp * c8 * np.exp(0.25 * rng.normal(size=n)), zc=rng.normal(size=n), zp=rng.normal(size=n),
              labc=np.ones(n), labp=np.ones(n), rv=np.ones(n))
    D = CM.Data(**kw)
    lv = RL.Lv(np.full(n, pi), np.log(c8), np.zeros(n), np.ones(n))
    V = RL.view(D, lv, np.arange(n))
    F = torch.exp(RL.ln_f(RL.lnc_of(torch.as_tensor(np.asarray(ths, float)), V, "CK"), V)).numpy()
    Z = np.exp(sig * rng.normal(size=n) - sig ** 2 / 2) * np.where(rng.random(n) < jump, 30.0, 1.0) / (1.0 + 29.0 * jump)
    D.rv = F * Z
    return D, lv


def fit_pair(D, lv, lam: float, mode: str) -> np.ndarray:
    V = RL.view(D, lv, np.arange(len(D.rv)))
    c, m = RL.clip_stats(RL.x0_of(V, "CK"), np.arange(len(D.rv)))
    return RL.fit_r(V, "CK", lam, c, m, mode)[0]


def oos(D, lv, th) -> float:
    V = RL.view(D, lv, np.arange(len(D.rv)))
    return float(RL.ell(V.rv.numpy(), RL.ln_f(RL.lnc_of(torch.as_tensor(th), V, "CK"), V).numpy()).mean())


def synth_null(reps: int = 20) -> tuple:
    out, nz = {}, {}
    for lam in RL.GRID:
        r = []
        for s in range(reps):
            D, lv = synth_data(2000, s, np.zeros(6))
            Dt, lt = synth_data(2000, 1000 + s, np.zeros(6))
            th = fit_pair(D, lv, lam, "R")
            r.append((oos(Dt, lt, th) / oos(Dt, lt, np.zeros(6)) - 1.0, float(np.abs(th).max() > 1e-6)))
        out[lam], nz[lam] = float(np.mean([a for a, _ in r])), float(np.mean([b for _, b in r]))
    return out, nz


def synth() -> list:
    bad, null = [], synth_null()
    for lam in RL.GRID:
        print("(а) нулевой сигнал, λ = %g: QLIKE вне выборки R − θ=0 = %+.4f %%, доля θ ≠ 0 = %.2f" % (lam, 100 * null[0][lam], null[1][lam]))
    if null[0][RL.GRID[0]] > 1e-3:
        bad.append("(а) λ = 0,1: %+.4f %%" % (100 * null[0][RL.GRID[0]]))
    ths = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.3])
    est = np.array([fit_pair(*synth_data(2000, 500 + s, ths), 1e-3, "R") for s in range(20)])
    err = np.abs(np.median(est, axis=0) - ths)
    print("(б) сигнал γ_p = 0,5, θ_p1 = 0,3: |медиана θ̂ − θ*| (γ_c, γ_p, θ_c0, θ_p0, θ_c1, θ_p1) =", np.round(err, 3), "; критерий — наклоны θ_c1, θ_p1 (γ слабо определены: SD оценки γ_p ≈ 0,26, справочно)")
    if err[4:].max() > 0.1:
        bad.append("(б) %s" % err[4:])
    D, lv = synth_data(2000, 7, np.zeros(6))
    base = [fit_pair(D, lv, 1e-3, md) for md in ("R", "plain")]
    D.rv = D.rv.copy()
    V = RL.view(D, lv, np.arange(2000))
    D.rv[0] = 40.0 * float(D.rv[0]) / float(RL.x0_of(V, "CK")[0])
    mov = [np.linalg.norm(fit_pair(D, lv, 1e-3, md) - b) for md, b in zip(("R", "plain"), base)]
    print("(в) пара x = 40: сдвиг θ̂ R = %.4f, обычная QLIKE = %.4f, отношение %.3f" % (mov[0], mov[1], mov[0] / mov[1]))
    if mov[0] > 0.2 * mov[1]:
        bad.append("(в) отношение %.3f" % (mov[0] / mov[1]))
    return bad
