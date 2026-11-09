"""V7 лист 1a-M, G17: обучение по QLIKE. Значение потерь = формула по слагаемым; градиент = численному (все четыре λ, MoE A/B/C, якорь, плоский, совместное с энкодером); ранняя остановка и решётка — по последним 126
закрытым парам; известный ответ: синтетика с двумя режимами — маршрутизатор восстанавливает режим, MoE не хуже плоского корректора в пределах объявленного допуска. Синтетика, данных SBER нет."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from models_v2 import metrics
from models_v2.s14 import v7_1m_fit as FT, v7_1m_forms as FM, v7_1m_loss as LS


def rn(*shape, seed=0):
    return torch.randn(tuple(shape), generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def small_batch(n=40, zdim=4):
    H = torch.exp(0.3 * rn(n, seed=4))
    return dict(Zg=rn(n, 9, seed=1), z=rn(n, zdim, seed=2), ell=0.3 * rn(n, seed=3), H=H, rv=H * torch.exp(0.4 * rn(n, seed=5)))


def randomize(model, seed=11):
    with torch.no_grad():
        for i, p in enumerate(model.parameters()):
            p.copy_(0.5 * rn(*p.shape, seed=seed + i))


def num_grad(model, fn, eps=1e-6):
    ps = list(model.parameters())
    out = []
    for p in ps:
        g = torch.zeros_like(p)
        flat, gf = p.data.view(-1), g.view(-1)
        for i in range(flat.numel()):
            old = float(flat[i])
            flat[i] = old + eps
            up = float(fn())
            flat[i] = old - eps
            dn = float(fn())
            flat[i] = old
            gf[i] = (up - dn) / (2 * eps)
        out.append(g)
    return out


def grad_gap(model, b, lg=0.7, lb=0.3):
    f = lambda: LS.loss(model, b, lg, lb)
    model.zero_grad()
    f().backward()
    an = [p.grad.clone() for p in model.parameters()]
    nu = num_grad(model, f)
    return max(float((a - n).abs().max()) for a, n in zip(an, nu))


@pytest.mark.parametrize("name,mk", [("A", lambda: FM.ResMoE("A")), ("B", lambda: FM.ResMoE("B")), ("C", lambda: FM.ResMoE("C")), ("anchor", lambda: FM.ResMoE("B", anchor=True)), ("flat", lambda: FM.Flat())])
def test_gradient_equals_numerical_all_regularizers(name, mk):
    m = mk()
    randomize(m)
    assert grad_gap(m, small_batch()) < 1e-6


def test_gradient_numerical_joint_with_encoder_penalty():
    enc = nn.Sequential(nn.Linear(6, 4)).double()
    j = LS.Joint(FM.ResMoE("B"), enc, lam_e=0.5)
    randomize(j.head)
    with torch.no_grad():
        for p in enc.parameters():
            p.add_(0.3 * rn(*p.shape, seed=21))                          # φ ≠ φ₀: штраф λ_e‖φ − φ₀‖² ненулевой
    b = small_batch(zdim=6)
    assert float(j.extra_penalty()) > 0
    assert grad_gap(j, b) < 1e-6


def test_mutation_detached_data_term_caught_by_gradient_check(monkeypatch):
    orig = LS.data_term
    monkeypatch.setattr(LS, "data_term", lambda H, rv, G: orig(H, rv, G.detach()))
    m = FM.ResMoE("B")
    randomize(m)
    assert grad_gap(m, small_batch()) > 1e-4


def _ref_loss(model, b, lg, lb, reg):
    """Формула плана на numpy по выходам модели."""
    with torch.no_grad():
        G, pi = model(b["Zg"], b["z"], b["ell"])
    G, H, rv = G.numpy(), b["H"].numpy(), b["rv"].numpy()
    F = H * np.exp(G)
    L = np.mean(np.log(F) + rv / F) + lg * np.mean(G ** 2) + reg["lam2"] * sum(float((p.detach().numpy() ** 2).sum()) for p in model.theta_params())
    pi = pi.numpy()
    L += lb * np.sum((pi.mean(axis=0) - 1.0 / pi.shape[1]) ** 2) + reg["lam_s"] * np.mean(np.sum((pi[1:] - pi[:-1]) ** 2, axis=1))
    return L


@pytest.mark.parametrize("reg", [dict(lam2=0.0, lam_s=0.0, lam_e=0.0), dict(lam2=0.01, lam_s=0.0, lam_e=0.0), dict(lam2=0.0, lam_s=0.5, lam_e=0.0), dict(lam2=0.003, lam_s=0.02, lam_e=0.0)])
@pytest.mark.parametrize("lg,lb", [(0.0, 0.0), (1.0, 0.0), (0.0, 0.4), (0.1, 0.1)])
def test_loss_value_equals_plan_formula_term_by_term(reg, lg, lb):
    m = FM.ResMoE("B")
    randomize(m)
    b = small_batch()
    assert float(LS.loss(m, b, lg, lb, reg)) == pytest.approx(_ref_loss(m, b, lg, lb, reg), rel=1e-12)


def test_data_term_is_qlike_up_to_constant():
    b = small_batch()
    G = 0.2 * rn(40, seed=8)
    q = float(torch.mean(torch.as_tensor(metrics.qlike(b["rv"].numpy(), (b["H"] * torch.exp(G)).numpy()))))
    const = float(torch.mean(torch.log(b["rv"]) + 1.0))
    assert float(LS.data_term(b["H"], b["rv"], G)) == pytest.approx(q + const, rel=1e-12)


def test_reg_constants_declared_in_one_place():
    assert set(LS.REG) == {"lam2", "lam_s", "lam_e"} and all(v > 0 for v in LS.REG.values())
    assert FT.GRID == ((0.1, 0.0), (0.1, 0.1), (1.0, 0.0), (1.0, 0.1)) and FT.SEEDS == (0, 1, 2, 3, 4)


def test_joint_encoder_penalty_pulls_to_pretrained_weights():
    def run(lam_e):
        torch.manual_seed(0)
        enc = nn.Linear(6, 4)
        spec = dict(kind="moe", router="B", encoder=lambda: enc, lam_e=lam_e)
        b = small_batch(120, zdim=6)
        m, _, _ = FT.train_one(spec, 0, 0.1, 0.0, b, None, 150)
        return float(sum(((p - p0) ** 2).sum() for p, p0 in zip(m.encoder.parameters(), m.phi0)))
    assert run(0.0) > 1e-4 and run(1e3) < 0.01 * run(0.0)


def test_joint_zero_at_init_gives_F_equals_H():
    spec = dict(kind="moe", router="B", encoder=lambda: nn.Linear(6, 4))
    m = FT.make(spec, 0)
    b = small_batch(30, zdim=6)
    assert torch.equal(b["H"] * torch.exp(m(b["Zg"], b["z"], b["ell"])[0]), b["H"])
