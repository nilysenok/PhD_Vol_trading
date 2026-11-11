"""S-R-IMPL-1 (Ж-254 пп. 6, 10–12; `S_R_SPEC.md`, разд. 2.1–2.3): потеря QLIKE-Хьюбера на уровне итога F, PIT-квантиль обрезки, поправка уровня. Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD.
Код пилотов `a_*`, `cm_*` не правится: импорт forward, penalty, границ и стартов.
Публичное API:
- `Q` = 0,99; `GRID` = (1e-1, 1e-2, 1e-3) — λ сетки Λ⁺ (θ = 0 — вложенный контроль); `ARCH` = {"A": "A2", "CK": "CK"} — вариант архитектуры.
- `level_arrays(arch, D=None, x=None, w5=None) -> Lv`: на парах архитектуры (A — 2451 пара K1, CK — 2198 пар D_M) π = `x1.pi` дня d пары, β·g = `w5g.beta`·1[gap = 1, refit_date ≠ NaT] дня d, ln B_mix; у пар A без X1 (d вне D_M) π := 1, ln B := 0, βg := 0 — ln F = ln H_A (при θ = 0 это K1).
- `view(D, lv, sel)`: `D.view(sel)` + тензоры уровня; `ln_f(lnC, V) = (1 − π)·ln B + π·lnC + βg`; `lnc_of(v, V, arch, corr)`: ln C_A (A2) или ln(cfg8·𝓜_K) (CK, μ = 0).
- `x0_of(V_all, arch, corr) = RV/F(0)` (W5-G на D_M, K1 у пар A без X1); `clip_stats(x0, T) -> (c, m)`: c = np.quantile(x0[T], 0,99, linear), m = mean min(x0[T], c) — только по парам T.
- `rho(u, kappa)`: eᵘ − u − 1 при u ≤ ln κ, иначе (κ − 1 − ln κ) + (κ − 1)(u − ln κ); `loss_r(v, V, arch, lam, c, m, mode, corr)`: mode "R" — mean ρ_κ(ln(RV/(m·F))) + штраф архитектуры; "plain" — обычная QLIKE уровня F (m = 1, κ = ∞) + штраф.
- `fit_r(V, arch, lam, c, m, mode, seeds, corr) -> [θ]`: L-BFGS-B (maxiter 500, ftol 1e-10) со стартами и границами архитектуры; A2 — θ по каждому seed (ln C усредняется), CK — минимум потери по перезапускам (ничья — меньший seed).
- `predict_lnc(thetas, V, arch, corr)`: среднее ln C по θ; `ell(rv, lnF)`: обычная QLIKE (массив); `pool_of(arch, D, x, a)`: закрытые пары месяца (позиция a в окне M), по возрастанию t⁺.
- ↺ `loss_r` имеет доп. аргумент `shift` (диагностика балла уровня, в обучении 0).
Запуск: `python3 -W ignore -m models_v2.s18.r_loss --check-nest | --check-grad | --check-pit | --synth` (только чтение; код — `r_loss_chk.py`)."""
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from models_v2.s13.v6_1_wf import closed_rows
from models_v2.s16 import x1_w3_core as C
from models_v2.s16 import x1_ya_walk as YA
from models_v2.s18 import a_model as AM
from models_v2.s18 import cm_model as CM

Q, GRID, ARCH, SEEDS = 0.99, (1e-1, 1e-2, 1e-3), {"A": "A2", "CK": "CK"}, (0, 1, 2, 3, 4)
W5G = C.ROOT / "data/regen/harj_x1/W5/w5g.parquet"


class Lv:
    def __init__(self, pi, lnb, bg, has):
        self.pi, self.lnb, self.bg, self.has = (np.asarray(a, float) for a in (pi, lnb, bg, has))


def level_arrays(arch: str, D=None, x=None, w5=None) -> Lv:
    x = C.load() if x is None else x
    w5 = pd.read_parquet(W5G) if w5 is None else w5
    if not (w5["d"].to_numpy() == x["d"].to_numpy()).all():
        raise SystemExit("w5g и окно M: даты расходятся")
    on = (w5["gap"].to_numpy() == 1) & w5["refit_date"].notna().to_numpy()
    pi, lnb, bg = x["pi"].to_numpy(float), np.log(x["M"].to_numpy(float)), np.where(on, w5["beta"].to_numpy(float), 0.0)
    if arch == "CK":
        return Lv(pi, lnb, bg, np.ones(len(x)))
    p = pd.Series(np.arange(len(x)), index=pd.DatetimeIndex(x["d"])).reindex(pd.DatetimeIndex(D.d)).to_numpy()
    has = ~np.isnan(p)
    j = np.where(has, p, 0).astype(int)
    return Lv(np.where(has, pi[j], 1.0), np.where(has, lnb[j], 0.0), np.where(has, bg[j], 0.0), has)


def view(D, lv: Lv, sel):
    V = D.view(sel)
    t = lambda a: torch.as_tensor(np.asarray(a, float)[V.sel])
    V.l_pi, V.l_lnb, V.l_bg = t(lv.pi), t(lv.lnb), t(lv.bg)
    return V


def ln_f(lnc, V):
    return (1.0 - V.l_pi) * V.l_lnb + V.l_pi * lnc + V.l_bg


def lnc_of(v, V, arch: str, corr: bool = True):
    if arch == "A":
        return AM.forward(v, V, "A2", corr)[1]
    return torch.log(V.c8 * CM.forward(v, V, "CK")["M"])


def n_par(arch: str) -> int:
    return AM.NP if arch == "A" else CM.NP


def x0_of(V_all, arch: str, corr: bool = True) -> np.ndarray:
    lnf = ln_f(lnc_of(torch.zeros(n_par(arch), dtype=torch.float64), V_all, arch, corr), V_all)
    return (V_all.rv * torch.exp(-lnf)).numpy()


def clip_stats(x0, T) -> tuple:
    p = np.asarray(x0, float)[np.asarray(T)]
    c = float(np.quantile(p, Q))
    return c, float(np.mean(np.minimum(p, c)))


def rho(u, kappa: float):
    lk = math.log(kappa)
    body = torch.exp(torch.clamp(u, max=lk)) - u - 1.0
    tail = (kappa - 1.0 - lk) + (kappa - 1.0) * (u - lk)
    return torch.where(u <= lk, body, tail)


def penalty_of(v, arch: str, lam: float):
    return AM.penalty(v, lam) if arch == "A" else CM.penalty(v, lam)


def pool_of(arch: str, D, x, a: int) -> np.ndarray:
    """Закрытые пары на дату переоценки x.d[a] (t⁺ ≤ D, `closed_rows` / `closed_pool` прежних проходов), по возрастанию t⁺."""
    xd = pd.DatetimeIndex(x["d"])
    if arch == "A":
        p, tp = closed_rows(D.tp, xd[a]), np.asarray(D.tp, "datetime64[ns]")
    else:
        tp = np.asarray(x["t_plus"], "datetime64[ns]")
        p = YA.closed_pool(tp, xd, xd[a])
    return p[np.argsort(tp[p], kind="stable")]


def loss_r(v, V, arch: str, lam: float, c: float, m: float, mode: str = "R", corr: bool = True, shift=0.0):
    """`shift` — диагностический сдвиг ln F (для балла уровня); в обучении 0."""
    u = torch.log(V.rv) - ln_f(lnc_of(v, V, arch, corr), V) - shift
    if mode == "R":
        r = rho(u - math.log(m), c / m)
    elif mode == "plain":
        r = torch.exp(u) - u - 1.0
    else:
        raise ValueError(mode)
    return r.mean() + penalty_of(v, arch, lam)


def _starts(V, arch: str, seeds, corr: bool) -> tuple:
    if arch == "A":
        free = AM.free_mask("A2", corr)
        return [AM._x0("A2", s, free) for s in seeds], AM._bounds(V, free)
    return [CM._x0("CK", s) for s in seeds], CM.bounds("CK")


def fit_r(V, arch: str, lam: float, c: float, m: float, mode: str = "R", seeds=SEEDS, corr: bool = True, maxiter: int = 500) -> list:
    def fun(x):
        v = torch.as_tensor(x).clone().requires_grad_(True)
        f = loss_r(v, V, arch, lam, c, m, mode, corr)
        f.backward()
        return float(f), v.grad.numpy().astype(float)
    x0s, bnd = _starts(V, arch, seeds, corr)
    runs = [minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bnd, options=dict(maxiter=maxiter, ftol=1e-10)) for x0 in x0s]
    if arch == "A":
        return [r.x for r in runs]
    return [runs[int(np.argmin([r.fun for r in runs]))].x]


def predict_lnc(thetas, V, arch: str, corr: bool = True) -> np.ndarray:
    return np.mean([lnc_of(torch.as_tensor(t), V, arch, corr).numpy() for t in thetas], axis=0)


def ell(rv, lnf) -> np.ndarray:
    rv, lnf = np.asarray(rv, float), np.asarray(lnf, float)
    return rv * np.exp(-lnf) + lnf - np.log(rv) - 1.0


if __name__ == "__main__":
    from models_v2.s18 import r_loss_chk
    sys.exit(r_loss_chk.main(sys.argv[1:]))
