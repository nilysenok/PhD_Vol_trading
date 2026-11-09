"""X1-пилот, лист P0, G3 (синтетика): выход из нулевой инициализации у ШАБЛОНОВ планируемых поправок трёх механизмов Ж-241. Планы поправок ещё не утверждены (лист P — после P0): формы ниже — минимальные носители механизма, не конфигурации; обучения нет.
Цель везде — потеря V7 (ln F + RV/F, torch float64) и F = X1·exp(G); при выключении (нулевые параметры) F = X1 побитно. Для каждого параметра — норма градиента в точке инициализации и пояснение: «сразу» или «после параметра …».
 М1 (состояние к закрытию и затухание): G = a·Σ_{j<L} exp(−κ·j)·s_{d−j}; a = 0.
 М2 (обучаемая агрегация M10): G = γ·ln(B(θ)/RV_d), B = Σ_k softmax(θ)_k·n·r²_k (при θ = 0 и w ≡ 1 B = RV_d); γ = 0; два варианта: θ = 0 и θ малый случайный.
 М3 (согласованные компоненты): F = C_x·exp(g_c) + J_x·exp(g_j), C_x + J_x = X1, g_c = a_c + b_c·u, g_j = a_j + b_j·u; все 0."""
from __future__ import annotations

import numpy as np
import torch

L_DECAY, N_SLOTS, N_DAYS = 5, 20, 1500


def data(seed: int = 0) -> dict:
    """Синтетика с ИЗВЕСТНОЙ связью: RV = X1·exp(0.3·s_sm + 0.2·(ln B_w − ln RV_d) + шум); все три сигнала настоящие, поэтому у «живых» параметров градиент обязан быть ненулевым."""
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)
    r2 = rn(N_DAYS, N_SLOTS) ** 2 * (1.0 + torch.arange(N_SLOTS, dtype=torch.float64) / N_SLOTS)
    rv = r2.sum(1)
    s = rn(N_DAYS + L_DECAY)
    sm = sum(np.exp(-0.7 * j) * s[L_DECAY - j:N_DAYS + L_DECAY - j] for j in range(L_DECAY))
    w_true = torch.softmax(torch.linspace(0.0, 1.0, N_SLOTS, dtype=torch.float64), 0)
    ln_ratio = torch.log((w_true * N_SLOTS * r2).sum(1) / rv)
    x1 = torch.exp(rn(N_DAYS))
    y = x1 * torch.exp(0.3 * sm + 0.2 * ln_ratio + 0.1 * rn(N_DAYS))
    return dict(r2=r2, rv=rv, s=s, x1=x1, y=y, u=rn(N_DAYS), jsh=0.2 + 0.1 * torch.rand(N_DAYS, generator=g, dtype=torch.float64))


def qlike_term(y, F):
    return torch.mean(torch.log(F) + y / F)


def form_m1(p: dict, d: dict):
    sm = sum(torch.exp(-p["kappa"] * j) * d["s"][L_DECAY - j:N_DAYS + L_DECAY - j] for j in range(L_DECAY))
    return d["x1"] * torch.exp(p["a"] * sm)


def form_m2(p: dict, d: dict):
    B = (torch.softmax(p["theta"], 0) * N_SLOTS * d["r2"]).sum(1)
    return d["x1"] * torch.exp(p["gamma"] * torch.log(B / d["rv"]))


def form_m3(p: dict, d: dict):
    cx, jx = d["x1"] * (1 - d["jsh"]), d["x1"] * d["jsh"]                     # C_x + J_x = X1; запись через expm1: при нулевых параметрах F = X1 побитно
    return d["x1"] + cx * torch.expm1(p["a_c"] + p["b_c"] * d["u"]) + jx * torch.expm1(p["a_j"] + p["b_j"] * d["u"])


def init_params(form: str, theta_std: float = 0.0, seed: int = 0) -> dict:
    z = lambda: torch.zeros((), dtype=torch.float64, requires_grad=True)
    if form == "m1":
        return dict(a=z(), kappa=torch.tensor(0.5, dtype=torch.float64, requires_grad=True))
    if form == "m2":
        th = theta_std * torch.randn(N_SLOTS, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
        return dict(gamma=z(), theta=th.requires_grad_())
    return dict(a_c=z(), b_c=z(), a_j=z(), b_j=z())


FORMS = {"m1": form_m1, "m2": form_m2, "m3": form_m3}


def grad_norms(form: str, theta_std: float = 0.0, d: dict | None = None) -> dict:
    """-> {параметр: ‖∂L/∂p‖} в точке инициализации + F(init) == X1 побитно (если параметры нулевые)."""
    d = d or data()
    p = init_params(form, theta_std)
    F = FORMS[form](p, d)
    qlike_term(d["y"], F).backward()
    out = {k: float(v.grad.norm()) for k, v in p.items()}
    out["_F_equals_X1"] = float(torch.equal(F.detach(), d["x1"]) if all(float(v.abs().sum()) == 0 for v in p.values()) else float("nan"))
    return out


def second_step_unblocks(form: str, theta_std: float = 0.0, step: float = 0.05) -> dict:
    """Один шаг градиентного спуска по параметрам с ненулевым градиентом, затем градиенты заново: показывает, какие параметры «оживают» только после другого (граф зависимости)."""
    d = data()
    p = init_params(form, theta_std)
    qlike_term(d["y"], FORMS[form](p, d)).backward()
    with torch.no_grad():
        for v in p.values():
            v -= step * v.grad
    for v in p.values():
        v.grad = None
    qlike_term(d["y"], FORMS[form](p, d)).backward()
    return {k: float(v.grad.norm()) for k, v in p.items()}


def rows() -> list:
    out = []
    for form, std, label in (("m1", 0.0, "М1: состояние и затухание"), ("m2", 0.0, "М2: агрегация, θ=0"), ("m2", 0.1, "М2: агрегация, θ~N(0,0.1²)"), ("m3", 0.0, "М3: компоненты C+J")):
        g0, g1 = grad_norms(form, std), second_step_unblocks(form, std)
        for k in init_params(form, std):
            out.append(dict(stage="synthetic_init", model=label, group=k, seed=-1, grad_data=g0[k], grad_pen=np.nan, grad_total=g0[k], num_vs_auto=np.nan, extra=g1[k],
                            note=("оживает только после другого параметра" if g1[k] > 0 else "МЁРТВАЯ ТОЧКА: градиент 0 и после шага (седло)") if g0[k] == 0 else "ненулевой сразу"))
    return out
