"""S-Z-IMPL-1 (Ж-258; `docs/otchety/harj_struct/S_Z_SPEC.md`, разд. 2, 3.4, 6, 8): линейная поправка ln C по z энкодера M10 под процедурой R — модель, потеря, подгонка головы, прогноз по пяти головам.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Код `r_*`, `cm_*`, `l_*`, `a_*`, `d_*` не правится: импорт `cm_model.forward / bounds / _x0 / penalty`, `r_loss.rho / ln_f`. Новых библиотек нет (torch autograd float64, scipy L-BFGS-B).
Публичное API:
- `CLIP` = 3,0 (↺ разд. 2.3: обрезка z̃; откат — `CLIP = None`); `NZ` = 4; `NP` = {"CKZ": 10 (θ_CK 6 + θ_Z 4), "Z": 4}; `ZB` = (−3; 3) — границы θ_Z.
- `std_stats(z, T) -> (μ, σ)`: по компонентам, ddof = 0, только строки T; нефинитные z в T или σ ≤ 0 — ValueError (подстановок нет, Д4).
- `ztil_raw(z, μ, σ)` = (z − μ)/σ; `ztil(z, μ, σ)` = clip(ztil_raw, −CLIP, CLIP) (после стандартизации; NaN остаётся NaN).
- `eta_of(zt, θ_Z)` = z̃ · θ_Z; `lnc_head(v, V, zt, var) -> tensor`: var "CKZ": ln C = log(c8·𝓜_CK(v[:6])) + η(v[6:]); var "Z": ln C = log(c8) + η(v) — η прибавляется ПОСЛЕ логарифма (при θ_Z = 0 путь `r_loss.lnc_of` побитно).
- `loss_z(v, V, zt, var, lam, c, m, mode="R")`: mean ρ_κ(ln(RV/(m·F))), κ = c/m, ln F = `r_loss.ln_f` (π, β·g замороженные, день пары), mode "plain" — обычная QLIKE уровня F; штраф λ·‖v‖² на весь вектор (`cm_model.penalty`).
- `starts(var, seeds=r_loss.SEEDS)`, `bounds(var)`: CKZ — θ_CK = `cm_model._x0("CK", r)`, θ_Z = 0, границы C-K и ±3; Z — один старт 0.
- `fit_head(V, zt, var, lam, c, m, mode="R", seeds=r_loss.SEEDS, maxiter=500) -> θ`: L-BFGS-B (ftol 1e-10), минимум потери по перезапускам, при равенстве меньший r.
- `predict_lnc(heads, V, zts, var=None) -> ndarray`: ln C = ln C^(0) + среднее (ln C^(s) − ln C^(0)) по головам (голова 0 + среднее отклонений; var по длине θ, если не задан).
Запуск: `python3 -W ignore -m models_v2.s18.z_model --check-nest | --check-grad | --check-pit | --synth` (только чтение; код — `z_model_chk.py`)."""
from __future__ import annotations

import torch  # noqa: F401  (первым, класс № 152)

import math
import sys

import numpy as np
from scipy.optimize import minimize

from models_v2.s18 import cm_model as CM
from models_v2.s18 import r_loss as RL

torch.set_num_threads(1)
CLIP, NZ, ZB = 3.0, 4, (-3.0, 3.0)
NP = {"CKZ": CM.NP + NZ, "Z": NZ}


def std_stats(z, T) -> tuple:
    z = np.asarray(z, float)[np.asarray(T)]
    if not np.isfinite(z).all():
        raise ValueError("std_stats: нефинитный z в T (подстановки запрещены, Д4)")
    mu, sd = z.mean(axis=0), z.std(axis=0)
    if (sd <= 0).any():
        raise ValueError("std_stats: σ ≤ 0")
    return mu, sd


def ztil_raw(z, mu, sd) -> np.ndarray:
    return (np.asarray(z, float) - mu) / sd


def ztil(z, mu, sd) -> np.ndarray:
    t = ztil_raw(z, mu, sd)
    return t if CLIP is None else np.clip(t, -CLIP, CLIP)


def eta_of(zt, thz):
    return torch.as_tensor(zt, dtype=torch.float64) @ thz


def lnc_head(v, V, zt, var: str):
    v = torch.as_tensor(v, dtype=torch.float64)
    if var not in NP or v.shape[0] != NP[var]:
        raise ValueError("lnc_head: var %r, длина θ %d" % (var, v.shape[0]))
    if var == "CKZ":
        return torch.log(V.c8 * CM.forward(v[:CM.NP], V, "CK")["M"]) + eta_of(zt, v[CM.NP:])
    return torch.log(V.c8) + eta_of(zt, v)


def loss_z(v, V, zt, var: str, lam: float, c: float, m: float, mode: str = "R"):
    u = torch.log(V.rv) - RL.ln_f(lnc_head(v, V, zt, var), V)
    if mode == "R":
        r = RL.rho(u - math.log(m), c / m)
    elif mode == "plain":
        r = torch.exp(u) - u - 1.0
    else:
        raise ValueError(mode)
    return r.mean() + CM.penalty(torch.as_tensor(v), lam)


def starts(var: str, seeds=RL.SEEDS) -> list:
    return [np.concatenate([CM._x0("CK", r), np.zeros(NZ)]) for r in seeds] if var == "CKZ" else [np.zeros(NZ)]


def bounds(var: str) -> list:
    return (CM.bounds("CK") if var == "CKZ" else []) + [ZB] * NZ


def fit_head(V, zt, var: str, lam: float, c: float, m: float, mode: str = "R", seeds=RL.SEEDS, maxiter: int = 500) -> np.ndarray:
    def fun(x):
        v = torch.as_tensor(x).clone().requires_grad_(True)
        f = loss_z(v, V, zt, var, lam, c, m, mode)
        f.backward()
        return float(f), v.grad.numpy().astype(float)
    runs = [minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds(var), options=dict(maxiter=maxiter, ftol=1e-10)) for x0 in starts(var, seeds)]
    return runs[int(np.argmin([r.fun for r in runs]))].x


def predict_lnc(heads, V, zts, var: str | None = None) -> np.ndarray:
    if len(heads) != len(zts) or not len(heads):
        raise ValueError("predict_lnc: головы и z̃ не совпадают по числу")
    var = var or ("CKZ" if len(heads[0]) == NP["CKZ"] else "Z")
    ls = [lnc_head(torch.as_tensor(h), V, zt, var).numpy() for h, zt in zip(heads, zts)]
    return ls[0] + np.mean([l - ls[0] for l in ls], axis=0)


if __name__ == "__main__":
    from models_v2.s18 import z_model_chk
    sys.exit(z_model_chk.main(sys.argv[1:]))
