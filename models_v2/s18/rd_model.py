"""S-RCD-IMPL-1 (Ж-259; `S_RCD_SPEC.md`, разд. 2–4, 8.2): модель и потеря уровня F для двух форм C — прямой C-D и компонентной C-K. Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD.
Код `r_*`, `cm_*` не правится: импорт forward, penalty, границ, стартов, ρ_κ, ln F. Форма передаётся аргументом (в `r_loss` форма C-K зашита).
Публичное API:
- `FORMS = ("CD", "CK")`; `lnc_of(v, V, form)` = ln(cfg8·𝓜_form(v)); `starts(form, seeds)` = [`CM._x0(form, s)`]; `bounds(form)` = `CM.bounds(form)`.
- `loss(v, V, form, lam, c, m, mode, shift=0.0)`: u = ln RV − ln F(ln C) − shift; mode "R" — mean ρ_κ(u − ln m), κ = c/m; "plain" — mean(eᵘ − u − 1) (обрезки нет, m = 1) + `CM.penalty` (все 6 параметров к нулю).
  Арифметика `r_loss.loss_r` для CK побитно; `shift` — диагностика балла уровня, в обучении 0.
- `fit_full(V, form, lam, c, m, mode, seeds=RL.SEEDS) -> ([θ], spread)`: L-BFGS-B (maxiter 500, ftol 1e-10), минимум штрафованной L по перезапускам, при равенстве меньший seed; spread = max L − min L по перезапускам.
  `fit(...) -> [θ]` (список из одного θ, как `RL.fit_r` у CK); `predict_lnc(thetas, V, form)` = среднее ln C по θ.
Запуск: `python3 -W ignore -m models_v2.s18.rd_model --check-nest | --check-grad | --check-pit | --synth` (только чтение; код — `rd_model_chk.py`, `rd_model_syn.py`)."""
from __future__ import annotations

import math
import sys

import numpy as np
import torch  # noqa: F401  (torch раньше lightgbm / xgboost — класс № 152)
from scipy.optimize import minimize

from models_v2.s18 import cm_model as CM
from models_v2.s18 import r_loss as RL

torch.set_num_threads(1)
FORMS = ("CD", "CK")
MODES = ("R", "plain")


def _form(form: str) -> str:
    if form not in FORMS:
        raise ValueError("форма %r: допустимы %s" % (form, FORMS))
    return form


def lnc_of(v, V, form: str):
    return torch.log(V.c8 * CM.forward(v, V, _form(form))["M"])


def starts(form: str, seeds=RL.SEEDS) -> list:
    return [CM._x0(_form(form), s) for s in seeds]


def bounds(form: str) -> list:
    return CM.bounds(_form(form))


def loss(v, V, form: str, lam: float, c: float, m: float, mode: str = "R", shift=0.0):
    u = torch.log(V.rv) - RL.ln_f(lnc_of(v, V, form), V) - shift
    if mode == "R":
        r = RL.rho(u - math.log(m), c / m)
    elif mode == "plain":
        r = torch.exp(u) - u - 1.0
    else:
        raise ValueError(mode)
    return r.mean() + CM.penalty(v, lam)


def fit_full(V, form: str, lam: float, c: float, m: float, mode: str = "R", seeds=RL.SEEDS, maxiter: int = 500) -> tuple:
    def fun(x):
        v = torch.as_tensor(x).clone().requires_grad_(True)
        f = loss(v, V, form, lam, c, m, mode)
        f.backward()
        return float(f), v.grad.numpy().astype(float)
    runs = [minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds(form), options=dict(maxiter=maxiter, ftol=1e-10)) for x0 in starts(form, seeds)]
    ls = np.array([r.fun for r in runs])
    return [runs[int(np.argmin(ls))].x], float(np.ptp(ls))


def fit(V, form: str, lam: float, c: float, m: float, mode: str = "R", seeds=RL.SEEDS, maxiter: int = 500) -> list:
    return fit_full(V, form, lam, c, m, mode, seeds, maxiter)[0]


def predict_lnc(thetas, V, form: str) -> np.ndarray:
    return np.mean([lnc_of(torch.as_tensor(t), V, form).numpy() for t in thetas], axis=0)


if __name__ == "__main__":
    from models_v2.s18 import rd_model_chk
    sys.exit(rd_model_chk.main(sys.argv[1:]))
