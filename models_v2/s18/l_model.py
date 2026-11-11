"""S-L-IMPL-1 (Ж-256 п. 5; `S_L_SPEC.md`, разд. 2, 3): сдвиг уровня C одним параметром a под процедурой R. C(a) = cfg8·e^a, a ∈ [−0,5; 0,5].
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD; исследовательский результат, не OOS. Код `r_*`, `cm_*`, `a_*`, `d_*` не правится; ρ — `r_loss.rho` без изменений.
Публичное API (массивы A — словарь numpy на 2198 парах: rv, lnb, pi, bg, lnc8):
- `A_BOUND` = 0,5; `lnc(a, lnc8) = lnc8 + a`; `ln_ftrain(a, lnb, pi_t, bg, lnc8) = (1 − π)·ln B + π·(ln cfg8 + a) + βg` (при a = 0 — W5-G на π обучения).
- `train_pi(kind, pi, T)`: "pi" → π[T]; "pibar" → π̄_T на все T; "shift252" → `np.roll(pi, 252)[T]` (циклически, В-4).
- `loss_l(a, rv, lnb, pi_t, bg, lnc8, lam, c, m)` (torch) = mean ρ_{c/m}(ln RV − ln F_train(a) − ln m) + λa².
- `stats_l(kind, A, T) -> (c, m)`: `clip_stats` по RV/F_train(0) того же вида (В-3).
- `fit_l(rv, lnb, pi_t, bg, lnc8, lam, c, m, start=0.0) -> float`: L-BFGS-B, границы ±A_BOUND, старт 0 (потеря выпукла по a, В-1), maxiter 500, аналитический градиент через torch.
Допуски L-BFGS-B ужесточены (↺): `FTOL` = 0 (остановка только по градиенту), `GTOL` = 1e-12 против умолчания 1e-10 / 1e-5 у R, иначе гейт L1.G5 (≤ 1e-8) недостижим при остановке по ftol."""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.optimize import minimize

from models_v2.s18 import r_loss as RL

A_BOUND, MAXITER, FTOL, GTOL, RESTARTS, POLISH, DH = 0.5, 500, 0.0, 1e-12, 4, 4, 1e-5
KINDS = ("pi", "pibar", "shift252")
KINDS_ALL = KINDS + ("shift252nr",)
SHIFT = 252


def lnc(a, lnc8):
    return lnc8 + a


def ln_ftrain(a, lnb, pi_t, bg, lnc8):
    return (1.0 - pi_t) * lnb + pi_t * lnc(a, lnc8) + bg


def train_pi(kind: str, pi, T) -> np.ndarray:
    pi, T = np.asarray(pi, float), np.asarray(T)
    if kind == "pi":
        return pi[T]
    if kind == "pibar":
        return np.full(len(T), float(pi[T].mean()))
    if kind == "shift252":
        return np.roll(pi, SHIFT)[T]
    if kind == "shift252nr":    # Ж-257: без переноса через конец; для i < 252 прошлого значения нет — отказ (такие пары исключает `train_idx`)
        if (T < SHIFT).any():
            raise ValueError("shift252nr: индекс < %d не имеет π на 252 строки назад" % SHIFT)
        return pi[T - SHIFT]
    raise ValueError(kind)


def train_idx(kind: str, T) -> np.ndarray:
    """Пары пула обучения: у shift252nr первые 252 строки исключены (не заполняются, Д4)."""
    T = np.asarray(T)
    return T[T >= SHIFT] if kind == "shift252nr" else T


def loss_l(a, rv, lnb, pi_t, bg, lnc8, lam: float, c: float, m: float):
    u = torch.log(rv) - ln_ftrain(a, lnb, pi_t, bg, lnc8)
    return RL.rho(u - math.log(m), c / m).mean() + lam * a ** 2


def stats_l(kind: str, A: dict, T) -> tuple:
    T = train_idx(kind, T)
    pi_t = train_pi(kind, A["pi"], T)
    x =A["rv"][T] * np.exp(-ln_ftrain(0.0, A["lnb"][T], pi_t, A["bg"][T], A["lnc8"][T]))
    return RL.clip_stats(x, np.arange(len(T)))


def fit_l(rv, lnb, pi_t, bg, lnc8, lam: float, c: float, m: float, start: float = 0.0) -> float:
    t = lambda z: torch.as_tensor(np.asarray(z, float))
    rv_, lnb_, pi_, bg_, lc_ = (t(z) for z in (rv, lnb, pi_t, bg, lnc8))

    def fun(x):
        a = torch.as_tensor(x).clone().requires_grad_(True)
        f = loss_l(a[0], rv_, lnb_, pi_, bg_, lc_, lam, c, m)
        f.backward()
        return float(f), a.grad.numpy().astype(float)
    x = np.array([start], float)
    for _ in range(RESTARTS):    # перезапуск с найденной точки сбрасывает приближение гессиана и доводит остановку по линейному поиску
        x1 = minimize(fun, x, jac=True, method="L-BFGS-B", bounds=[(-A_BOUND, A_BOUND)], options=dict(maxiter=MAXITER, ftol=FTOL, gtol=GTOL)).x
        done, x = bool(np.array_equal(x1, x)), x1
        if done:
            break
    for _ in range(POLISH):    # потеря выпукла по a, разности потери ниже шума float64: доводка секущей по аналитическому градиенту
        g, h = fun(x)[1][0], (fun(x + DH)[1][0] - fun(x - DH)[1][0]) / (2 * DH)
        if not (h > 0 and abs(g) > GTOL):
            break
        step = float(np.clip(x[0] - g / h, -A_BOUND, A_BOUND)) - x[0]
        x = x + step
        if abs(step) < 1e-15:
            break
    return float(np.clip(x[0], -A_BOUND, A_BOUND))


def fit_kind(kind: str, A: dict, T, lam: float, c: float, m: float, start: float = 0.0) -> float:
    T = train_idx(kind, T)
    return fit_l(A["rv"][T], A["lnb"][T], train_pi(kind, A["pi"], T), A["bg"][T], A["lnc8"][T], lam, c, m, start)
