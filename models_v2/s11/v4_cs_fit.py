"""V4-CS: оценщик C8 (два режима) с убывающими штрафами холодного старта (Ж-225 п. 3). Цель и градиент — как в `v4_cb_fit.fit_regime`, но штраф на отклонения по бумагам — вектор c_i = κ_i·s² (длина n),
панельный штраф — l = λ·s². Расписание: sched(n) = 250·250/(250 + n); при n = 0 равно 250 (C8). CS1: λ = sched(число закрытых пар панели), κ_i = 250; CS2: λ = 250, κ_i = sched(закрытые пары бумаги i).
Гиперпараметры фиксированы объявлением (GATES_V4.md, «Лист DX-CS»), не выбираются."""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from models_v2.s11 import v4_b_fit as FT, v4_c_fit as CF, v4_cb_fit as FB

K0 = float(FT.KAPPA)


def sched(n):
    """Сила приора / штрафа при n накопленных парах: 250·250/(250 + n); n = 0 даёт 250 (как в C8)."""
    return K0 * K0 / (K0 + np.asarray(n, float))


def fit_regime_vec(rv, lH, ell, z, tid, n, kappa_i, lam) -> dict:
    """C8: w = π·w_calm_i + (1 − π)·w_crisis_i, π = σ(a + b·z). kappa_i — вектор (n,) штрафов отклонений по бумагам в парах-эквивалентах, lam — скаляр для (a, b, a_calm, a_crisis).
    Старт тот же, что в `fit_regime` (a = 0, b = −1). -> {a, b, a_calm, a_crisis, d_calm, d_crisis, ok, grad, nit}."""
    s2 = float(np.mean(np.asarray(ell) ** 2))
    c, l = np.asarray(kappa_i, float) * s2, float(lam) * s2
    N = len(rv)

    def f(x):
        a, b, ac, ar, dc, dr = x[0], x[1], x[2], x[3], x[4:4 + n], x[4 + n:]
        p, wc, wr = expit(a + b * z), expit(ac + dc[tid]), expit(ar + dr[tid])
        w = p * wc + (1.0 - p) * wr
        lF = lH + w * ell
        e = rv * np.exp(-lF)
        gw = (1.0 - e) * ell
        gp = gw * (wc - wr) * p * (1.0 - p)
        gc, gr = gw * p * wc * (1.0 - wc), gw * (1.0 - p) * wr * (1.0 - wr)
        val = (np.sum(e + lF) + 0.5 * (float((c * dc) @ dc) + float((c * dr) @ dr)) + 0.5 * l * (a * a + b * b + ac * ac + ar * ar)) / N
        g = np.concatenate([[gp.sum() + l * a, (gp * z).sum() + l * b, gc.sum() + l * ac, gr.sum() + l * ar], np.bincount(tid, gc, minlength=n) + c * dc, np.bincount(tid, gr, minlength=n) + c * dr]) / N
        return val, g
    x0 = np.zeros(4 + 2 * n)
    x0[1] = -1.0
    lo, hi = np.full(len(x0), -CF.BOUND), np.full(len(x0), CF.BOUND)
    r, g = FB.lbfgs(f, x0, lo, hi)
    x = r.x
    return {"a": float(x[0]), "b": float(x[1]), "a_calm": float(x[2]), "a_crisis": float(x[3]), "d_calm": x[4:4 + n].copy(), "d_crisis": x[4 + n:].copy(), "ok": bool(g < CF.GTOL), "grad": g, "nit": int(r.nit)}


def penalties(cfg: str, n_panel: int, n_i) -> tuple:
    """-> (kappa_i (вектор по бумагам), lam) конфигурации по накопленным закрытым парам: CS1 — λ от n панели, CS2 — κ_i от n_i бумаги. Другая cfg — ValueError."""
    n_i = np.asarray(n_i, float)
    if cfg == "CS1":
        return np.full(len(n_i), K0), float(sched(n_panel))
    if cfg == "CS2":
        return sched(n_i), K0
    raise ValueError("неизвестная конфигурация Cold Start: %s" % cfg)
