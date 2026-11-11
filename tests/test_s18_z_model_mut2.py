"""S-Z-IMPL-1 (Ж-258, разд. 8): мутации `z_model` — π дня t⁺ вместо дня d пары; `detach` в ветке u > ln κ (сверка градиента). Подмена через `monkeypatch` (класс № 43 / 110)."""
import torch  # noqa: F401  (первым: конфликт OpenMP, класс № 152)

import math

import numpy as np
import pytest

from models_v2.s18 import cm_model as CM
from models_v2.s18 import r_loss as RL
from models_v2.s18 import z_model as ZM
from models_v2.s18.r_loss_syn import synth_data

HEAD = ZM.lnc_head


def world(n=300, seed=0, sig_z=0.0):
    rng = np.random.default_rng(seed)
    D, lv = synth_data(n, seed, np.zeros(6))
    z = rng.normal(2.0, 3.0, size=(n, 4))
    zt = ZM.ztil(z, *ZM.std_stats(z, np.arange(n)))
    lv = RL.Lv(rng.uniform(0.3, 1.0, n), lv.lnb, 0.1 * rng.normal(size=n), lv.has)
    D.rv = D.rv * np.exp(sig_z * zt[:, 0])
    return RL.view(D, lv, np.arange(n)), zt


# ---------- π дня пары ----------
def ok_pi() -> bool:
    V, zt = world(300, 4, 0.3)
    v = 0.3 * np.random.default_rng(3).normal(size=4)
    pi, lnb, bg = V.l_pi.numpy(), V.l_lnb.numpy(), V.l_bg.numpy()
    c, m = 2.2, 0.9
    u = np.log(V.rv.numpy()) - ((1 - pi) * lnb + pi * HEAD(v, V, zt, "Z").numpy() + bg)
    lk, w = math.log(c / m), u - math.log(m)
    r = np.where(w <= lk, np.exp(np.minimum(w, lk)) - w - 1.0, (c / m - 1.0 - lk) + (c / m - 1.0) * (w - lk))
    return abs(float(ZM.loss_z(v, V, zt, "Z", 0.0, c, m)) - r.mean()) < 1e-12


def test_pi_of_day_of_pair_honest_and_mutation_pi_of_t_plus_caught(monkeypatch):
    assert ok_pi()

    def ln_f_next(lnc, V):
        p = torch.roll(V.l_pi, -1)                  # π строки t⁺ (следующий день), а не дня d пары
        return (1.0 - p) * V.l_lnb + p * lnc + V.l_bg
    monkeypatch.setattr(RL, "ln_f", ln_f_next)
    assert not ok_pi()


# ---------- градиент ----------
def grad_err(var: str) -> tuple:
    V, zt = world(300, 5, 0.3)
    rng = np.random.default_rng(9)
    c, m = 1.6, 0.9
    x = np.concatenate([CM._x0("CK", 1), 0.3 * rng.normal(size=4)]) if var == "CKZ" else 0.3 * rng.normal(size=4)
    f = lambda a: ZM.loss_z(a, V, zt, var, 1e-2, c, m, "R")
    v = torch.as_tensor(x).clone().requires_grad_(True)
    f(v).backward()
    fd = np.array([(float(f(torch.as_tensor(x + h))) - float(f(torch.as_tensor(x - h)))) / 2e-6 for h in 1e-6 * np.eye(len(x))])
    g = v.grad.numpy()
    u = (torch.log(V.rv) - RL.ln_f(HEAD(torch.as_tensor(x), V, zt, var), V) - math.log(m)).detach().numpy()
    return float(np.max(np.abs(g - fd)) / max(np.max(np.abs(g)), 1e-8)), int((u > math.log(c / m)).sum())


@pytest.mark.parametrize("var", ["CKZ", "Z"])
def test_gradient_matches_finite_differences_honest(var):
    err, hi = grad_err(var)
    assert hi >= 2 and err < 1e-6


@pytest.mark.parametrize("var", ["CKZ", "Z"])
def test_mutation_detach_in_tail_branch_caught(monkeypatch, var):
    def rho_detach(u, kappa):
        lk = math.log(kappa)
        body = torch.exp(torch.clamp(u, max=lk)) - u - 1.0
        tail = (kappa - 1.0 - lk) + (kappa - 1.0) * (u.detach() - lk)
        return torch.where(u <= lk, body, tail)
    monkeypatch.setattr(RL, "rho", rho_detach)
    err, hi = grad_err(var)
    assert hi >= 2 and err > 1e-6
