"""S-C-IMPL-2 (Ж-254 пп. 9–12; `docs/otchety/harj_struct/S_C_SPEC.md`, разд. 1.4–1.5, 2, 3): множитель 𝓜_V компонентного гибрида C, функция потерь, штраф, L-BFGS-B с перезапусками.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Только расчёт; чтение данных — через `cm_feat` (признаки) и `x1_w3_core.load` (RV пар, cfg8). Новых библиотек нет (torch autograd float64, scipy).
Публичное API:
- `VARIANTS = ("CD", "CK", "CKA", "CKL", "C7")`; `TRAINED = VARIANTS[:4]`; `NP = 6`; `NAMES[var]` — имена 6 параметров: CD = (th0, rho_c, rho_p, th_c1, th_p1, th_s); CK/CKA/CKL = (gam_c, gam_p, th_c0, th_p0, th_c1, th_p1). C7 — без параметров.
- `MU = {"CD": 0, "CK": 0, "CKA": 0.5, "CKL": 0}` (μ компонентной потери, фиксировано до счёта), `LAMS = (1e-4, 1e-3, 1e-2)`, `SEEDS = (0..4)`, `SEED_SD = 0.05`, `bounds(var)` — γ, ρ ∈ [0, 1], прочие [−3, 3].
- `Data(**arrays)`: массивы на парах (d, t⁺): rv (RV_{t⁺}), c8 (cfg8_d), Hc, Hp (Ĥ_X,d), sc, sp, su (ŝ_X,d), zc = ln(c_d/c_m), zp, labc, labp (c_{t⁺}, p_{t⁺}); `Data.view(sel)` -> View (позиции пар). `build_data(x, feat, lag=False) -> Data`: x — окно M (`d, t_plus, rv, C`), feat — `cm_feat.build`; lag = True — колонки `L_*` (сдвинутое разложение S-CKL). Отказ при нефинитных входах.
- `forward(v, V, var) -> dict(M, Fc, Fp, Fu)` (torch): CK-класс: A_X = ŝ_X·cfg8; η_X = γ_X·ln(Ĥ_X/A_X) + θ_X0 + θ_X1·ln(X_d/X_m); F_X = A_X·exp(η_X); F_u = ŝ_u·cfg8; 𝓜 = 1 + Σ ŝ_X·expm1(η_X) (при v = 0 точно 1.0). CD: 𝓜 = exp(θ0 + ρ_c ln(Ĥ_c/A_c) + ρ_p ln(Ĥ_p/A_p) + θ_c1 zc + θ_p1 zp + θ_s·logit ŝ_c), F_c, F_p, F_u = NaN. C7 (`v` игнорируется): 𝓜 = (Ĥ_c + Ĥ_p + ŝ_u·cfg8)/cfg8, F_c = Ĥ_c, F_p = Ĥ_p.
- `qlike(rv, F)`: rv/F − ln(rv/F) − 1. `qlike_sum(v, V, var)` — среднее по парам QLIKE суммы cfg8·𝓜. `penalty(v, lam)` = λ·‖v‖² (все 6 параметров, к нулю, то есть к cfg8). `loss(v, V, var, lam, mu=None)` = qlike_sum + μ·½·mean[ℓ(c_{t⁺}, F_c) + ℓ(p_{t⁺}, F_p)] + penalty.
- `fit(V, var, lam, mu=None, seeds=SEEDS, maxiter=500) -> dict(theta (6), seed, loss, losses (по seeds))`: L-BFGS-B (ftol 1e-10), градиенты autograd; seed 0 — точный ноль, seeds 1–4 — ноль + N(0; 0,05²) (rng seed), γ, ρ обрезаны в [0, 1]; итог — перезапуск с минимумом штрафованной L, при равенстве меньший seed.
Запуск: `python3 -W ignore -m models_v2.s18.cm_model --check-nest | --check-grad | --synth` (только чтение; код — `cm_model_chk.py`)."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

VARIANTS = ("CD", "CK", "CKA", "CKL", "C7")
TRAINED, NP, SEED_SD = VARIANTS[:4], 6, 0.05
LAMS, SEEDS = (1e-4, 1e-3, 1e-2), (0, 1, 2, 3, 4)
MU = {"CD": 0.0, "CK": 0.0, "CKA": 0.5, "CKL": 0.0}
NAMES = {"CD": ("th0", "rho_c", "rho_p", "th_c1", "th_p1", "th_s")}
NAMES.update({k: ("gam_c", "gam_p", "th_c0", "th_p0", "th_c1", "th_p1") for k in ("CK", "CKA", "CKL")})
UNIT = {"CD": (1, 2), "CK": (0, 1), "CKA": (0, 1), "CKL": (0, 1)}     # индексы γ / ρ ∈ [0, 1]
FIELDS = ("rv", "c8", "Hc", "Hp", "sc", "sp", "su", "zc", "zp", "labc", "labp")


def bounds(var: str) -> list:
    return [(0.0, 1.0) if i in UNIT[var] else (-3.0, 3.0) for i in range(NP)]


class Data:
    def __init__(self, **kw):
        for k in FIELDS:
            setattr(self, k, np.asarray(kw[k], float))
        self.n = len(self.rv)

    def view(self, sel) -> "View":
        return View(self, sel)


class View:
    def __init__(self, D: Data, sel):
        sel = np.asarray(sel)
        t = lambda a: torch.as_tensor(np.asarray(a, float)[sel])
        self.sel = sel
        self.rv, self.c8, self.Hc, self.Hp, self.sc, self.sp, self.su = (t(getattr(D, k)) for k in FIELDS[:7])
        self.zc, self.zp, self.labc, self.labp = t(D.zc), t(D.zp), t(D.labc), t(D.labp)
        lnc8 = torch.log(self.c8)
        self.xc = torch.log(self.Hc) - torch.log(self.sc) - lnc8       # ln(Ĥ_c / A_c)
        self.xp = torch.log(self.Hp) - torch.log(self.sp) - lnc8
        self.logit = torch.log(self.sc / (1.0 - self.sc))


def build_data(x: pd.DataFrame, feat: pd.DataFrame, lag: bool = False) -> Data:
    pre = "L_" if lag else ""
    f = lambda k: feat[pre + k].to_numpy(float)
    i, j = feat.index.get_indexer(pd.DatetimeIndex(x["d"])), feat.index.get_indexer(pd.DatetimeIndex(x["t_plus"]))
    if (i < 0).any() or (j < 0).any():
        raise SystemExit("дата пары не на сетке сессий C")
    D = Data(rv=x["rv"], c8=x["C"], Hc=f("Hc")[i], Hp=f("Hp")[i], sc=f("sc")[i], sp=f("sp")[i], su=f("su")[i],
             zc=np.log(f("c")[i] / f("c_m")[i]), zp=np.log(f("p")[i] / f("p_m")[i]), labc=f("c")[j], labp=f("p")[j])
    for k in FIELDS:
        if not np.isfinite(getattr(D, k)).all():
            raise SystemExit("нефинитный вход %s%s" % (pre, k))
    return D


def forward(v, V: View, var: str) -> dict:
    nan = torch.full_like(V.rv, float("nan"))
    if var == "C7":
        return dict(M=(V.Hc + V.Hp + V.su * V.c8) / V.c8, Fc=V.Hc, Fp=V.Hp, Fu=V.su * V.c8)
    if var == "CD":
        lnM = v[0] + v[1] * V.xc + v[2] * V.xp + v[3] * V.zc + v[4] * V.zp + v[5] * V.logit
        return dict(M=torch.exp(lnM), Fc=nan, Fp=nan, Fu=nan)
    Ac, Ap = V.sc * V.c8, V.sp * V.c8
    ec, ep = v[0] * V.xc + v[2] + v[4] * V.zc, v[1] * V.xp + v[3] + v[5] * V.zp
    return dict(M=1.0 + V.sc * torch.expm1(ec) + V.sp * torch.expm1(ep), Fc=Ac * torch.exp(ec), Fp=Ap * torch.exp(ep), Fu=V.su * V.c8)


def qlike(rv, F):
    r = rv / F
    return r - torch.log(r) - 1.0


def qlike_sum(v, V: View, var: str):
    return qlike(V.rv, V.c8 * forward(v, V, var)["M"]).mean()


def penalty(v, lam: float):
    return lam * (v ** 2).sum()


def loss(v, V: View, var: str, lam: float, mu: float | None = None):
    mu = MU[var] if mu is None else mu
    f = forward(v, V, var)
    out = qlike(V.rv, V.c8 * f["M"]).mean()
    if mu:
        out = out + mu * 0.5 * (qlike(V.labc, f["Fc"]) + qlike(V.labp, f["Fp"])).mean()
    return out + penalty(v, lam)


def _x0(var: str, seed: int) -> np.ndarray:
    x = np.zeros(NP)
    if seed:
        x = np.random.default_rng(seed).normal(0.0, SEED_SD, NP)
        x[list(UNIT[var])] = np.clip(x[list(UNIT[var])], 0.0, 1.0)
    return x


def fit(V: View, var: str, lam: float, mu: float | None = None, seeds=SEEDS, maxiter: int = 500) -> dict:
    def fun(x):
        v = torch.as_tensor(x).clone().requires_grad_(True)
        f = loss(v, V, var, lam, mu)
        f.backward()
        return float(f), v.grad.numpy().astype(float)
    runs = [minimize(fun, _x0(var, s), jac=True, method="L-BFGS-B", bounds=bounds(var), options=dict(maxiter=maxiter, ftol=1e-10)) for s in seeds]
    ls = np.array([r.fun for r in runs])
    k = int(np.argmin(ls))
    return dict(theta=runs[k].x, seed=int(seeds[k]), loss=float(ls[k]), losses=ls)


if __name__ == "__main__":
    from models_v2.s18 import cm_model_chk
    sys.exit(cm_model_chk.main(sys.argv[1:]))
