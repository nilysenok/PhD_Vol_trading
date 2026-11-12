"""S-RCD-IMPL-1, тестировщик (Ж-259): синтетика и детекторы для `rd_model`. Детектор возвращает список замечаний: на честном коде пуст, на мутанте — нет. Без сети, боевых данных и записи."""
import copy
import math

import numpy as np
import torch

from models_v2.s18 import cm_model as CM
from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_loss_syn as SY
from models_v2.s18 import rd_model as RM

ZERO = np.zeros(6)


def syn(n: int = 1200, seed: int = 0, th=ZERO, form: str = "CD", sig: float = 0.5, jump: float = 0.01, pi: float = 0.5, jit: float = 0.2, lnb_shift: float = 0.0):
    """Реальная форма CM.forward; RV = F*(θ*, форма)·Z, E Z = 1. Возвращает (D, lv)."""
    D, lv0 = SY.synth_data(n, seed, ZERO, pi=pi, sig=sig, jump=jump)
    rng = np.random.default_rng(1000 + seed)      # π, β·g меняются по дням (иначе «π следующего дня» неотличим); jit = 0 — константы
    lv = RL.Lv(np.clip(pi + jit * rng.normal(size=n), 0.05, 0.95) if jit else np.full(n, pi), lv0.lnb + lnb_shift, 0.02 * rng.normal(size=n) * (jit > 0), lv0.has)
    V = D.view(np.arange(n))
    f = lambda t: np.exp(lnf_ref(lnc_ref(torch.as_tensor(np.asarray(t, float)), V, form).numpy(), lv, np.arange(n)))
    D2 = copy.copy(D)
    D2.rv = D.rv / f(ZERO) * f(th)
    return D2, lv


def lnc_ref(v, V, form: str):
    """Независимо от `rd_model`: ln(cfg8·𝓜_form) прямо из `cm_model.forward`."""
    return torch.log(V.c8 * CM.forward(v, V, form)["M"])


def lnf_ref(lnc, lv, sel):
    """Независимо от `r_loss.ln_f` и `r_loss.view`: (1 − π)·ln B + π·ln C + β·g по массивам `lv` строк sel."""
    sel = np.asarray(sel)
    return (1.0 - lv.pi[sel]) * lv.lnb[sel] + lv.pi[sel] * np.asarray(lnc, float) + lv.bg[sel]


def rand_theta(form: str, seed: int = 1) -> np.ndarray:
    lo, hi = np.array(CM.bounds(form)).T
    return np.clip(np.random.default_rng(seed).normal(0.0, 0.4, 6), lo, hi)


def x0_of(D, lv, form: str) -> np.ndarray:
    V = D.view(np.arange(len(lv.pi)))
    return V.rv.numpy() * np.exp(-lnf_ref(lnc_ref(torch.zeros(6, dtype=torch.float64), V, form).numpy(), lv, np.arange(len(lv.pi))))


def form_problems() -> list:
    """lnc_of: CD — побитно log(c8·CM.forward CD), CK — побитно RL.lnc_of; формы различаются; при θ = 0 совпадают (побитно)."""
    D, lv = syn()
    V, bad = RL.view(D, lv, np.arange(1200)), []
    for form in RM.FORMS:
        v = torch.as_tensor(rand_theta(form))
        bad += ["lnc_of %s ≠ log(c8·M)" % form] * (not torch.equal(RM.lnc_of(v, V, form), torch.log(V.c8 * CM.forward(v, V, form)["M"])))
    v = torch.as_tensor(np.array([0.3, 0.5, 0.4, -0.2, 0.3, 0.2]))
    bad += ["lnc_of CK ≠ RL.lnc_of"] * (not torch.equal(RM.lnc_of(v, V, "CK"), RL.lnc_of(v, V, "CK")))
    bad += ["формы CD и CK не различаются"] * torch.allclose(RM.lnc_of(v, V, "CD"), RM.lnc_of(v, V, "CK"))
    z = torch.zeros(6, dtype=torch.float64)
    bad += ["θ = 0: CD ≠ CK"] * (not torch.equal(RM.lnc_of(z, V, "CD"), RM.lnc_of(z, V, "CK")))
    return bad


def bounds_problems() -> list:
    """Границы формы CD — CM.bounds("CD") (θ0 ∈ [−3; 3]); CK — CM.bounds("CK"); известный ответ: сдвиг уровня −0,3 при plain восстанавливается θ̂0 ≈ −0,3 (с границами C-K θ0 ∈ [0; 1] недостижимо)."""
    bad = ["bounds CD"] * (RM.bounds("CD") != CM.bounds("CD")) + ["bounds CK"] * (RM.bounds("CK") != CM.bounds("CK"))
    bad += ["bounds CD = bounds CK"] * (RM.bounds("CD") == RM.bounds("CK"))
    est = []
    for s in range(10):     # медиана по 10 выборкам: одна выборка с скачками ×30 шумна (SD ≈ 0,1)
        D, lv = syn(1200, 30 + s, th=np.array([-0.3, 0, 0, 0, 0, 0]))
        est.append(RM.fit(RL.view(D, lv, np.arange(1200)), "CD", 1e-4, float("inf"), 1.0, "plain")[0][0])
    return bad + ["медиана θ̂0 = %.3f вместо −0,3" % np.median(est)] * bool(abs(np.median(est) + 0.3) > 0.1)


def plain_problems() -> list:
    """plain: L = mean(eᵘ − u − 1) + λ‖θ‖² на пуле с парой x = 40; c, m (конечные) игнорируются."""
    D, lv = syn(600, 5)
    D.rv = D.rv.copy()
    D.rv[10] = 40.0 * D.rv[10] / x0_of(D, lv, "CD")[10]     # F(0)[10] = rv/x0 ⇒ пара x = RV/F(0) = 40
    V, v = RL.view(D, lv, np.arange(600)), rand_theta("CD", 2)
    u = np.log(V.rv.numpy()) - lnf_ref(lnc_ref(torch.as_tensor(v), V, "CD").numpy(), lv, np.arange(600))
    want = float(np.mean(np.exp(u) - u - 1.0)) + 1e-2 * float((v ** 2).sum())
    got = float(RM.loss(torch.as_tensor(v), V, "CD", 1e-2, 2.0, 0.5, "plain"))
    return ["plain %.12f ≠ %.12f" % (got, want)] * (abs(got - want) > 1e-12)


def ell_np(u):
    return np.exp(u) - u - 1.0


def rho_problems() -> list:
    """R: независимый numpy-расчёт ρ_κ(u − ln m), κ = c/m, на U с парами выше ln κ; π, β из `lv` (π ≠ 1 — не уровень C)."""
    D, lv = syn(800, 6, pi=0.4)
    lv.bg = lv.bg + 0.05
    V, v, c, m = RL.view(D, lv, np.arange(800)), rand_theta("CD", 3), 1.6, 0.9
    lnf = lnf_ref(lnc_ref(torch.as_tensor(v), V, "CD").numpy(), lv, np.arange(800))
    w, k = np.log(V.rv.numpy()) - lnf - math.log(m), c / m
    r = np.where(w <= math.log(k), ell_np(w), (k - 1 - math.log(k)) + (k - 1) * (w - math.log(k)))
    want, got = float(r.mean()) + 1e-3 * float((v ** 2).sum()), float(RM.loss(torch.as_tensor(v), V, "CD", 1e-3, c, m, "R"))
    return ["R-потеря %.12f ≠ %.12f (пар выше ln κ: %d)" % (got, want, int((w > math.log(k)).sum()))] * bool(abs(got - want) > 1e-12 or (w > math.log(k)).sum() < 10)


def pit_problems() -> list:
    """Порча RV, признаков, π, β строк вне пула T не меняет c, m и L на T (обе формы). Порча включает строку T.max()+1 (π дня t⁺ вместо d)."""
    from models_v2.s18 import rd_model_chk as MC
    D, lv = syn(1000, 12, pi=0.4)
    n, T, bad = 1000, np.arange(700), []
    out = np.ones(n, bool)
    out[T] = False
    Dc, lc = MC.corrupt(D, lv, out)
    for form in RM.FORMS:
        v = torch.as_tensor(rand_theta(form))
        s0, s1 = RL.clip_stats(x0_of(D, lv, form), T), RL.clip_stats(x0_of(Dc, lc, form), T)
        l0, l1 = (float(RM.loss(v, RL.view(d_, l_, T), form, 1e-2, *s, "R")) for d_, l_, s in ((D, lv, s0), (Dc, lc, s1)))
        bad += ["PIT %s: c, m %s ≠ %s; L %.6e ≠ %.6e" % (form, s0, s1, l0, l1)] * (s0 != s1 or l0 != l1)
    return bad


def grad_problems(form: str) -> list:
    """autograd против центральных разностей (h = 1e-6) ≤ 1e-6 отн.; ≥ 2 точек с парами выше ln κ; ≥ 1 точка с ρ / γ на границе."""
    D, lv = syn(600, 7)
    V, bad, hi_pts, bnd = RL.view(D, lv, np.arange(600)), [], 0, 0
    pts = RM.starts(form, (1, 2, 3)) + [np.array([0.3, 1.0, 0.0, -0.2, 0.4, 0.1]), np.array([0.0, 0.5, 0.2, 0.1, -0.1, 0.2])]
    for p in pts:
        f = lambda v: RM.loss(v, V, form, 1e-2, 1.5, 1.0, "R")
        v = torch.as_tensor(p).clone().requires_grad_(True)
        f(v).backward()
        fd = np.array([(float(f(torch.as_tensor(p + h))) - float(f(torch.as_tensor(p - h)))) / 2e-6 for h in 1e-6 * np.eye(6)])
        err = float(np.max(np.abs(v.grad.numpy() - fd)) / max(np.max(np.abs(v.grad.numpy())), 1e-8))
        u = (torch.log(V.rv) - RL.ln_f(RM.lnc_of(torch.as_tensor(p), V, form), V)).numpy()
        hi_pts += int((u > math.log(1.5)).sum() > 0)
        bnd += int(any(p[i] in (0.0, 1.0) for i in CM.UNIT[form]))
        bad += ["градиент %s: %.2e" % (form, err)] * (err > 1e-6)
    return bad + ["точек выше ln κ: %d" % hi_pts] * (hi_pts < 2) + ["нет точки на границе"] * (bnd < 1)
