"""X1-пилот P0, G3 / Ж-242: градиент маршрутизатора MoE V7 при нулевых экспертах равен 0 (известный ответ), при ненулевых ≠ 0; автоградиент = центральная разность;
эксперты-копии при равномерном маршрутизаторе (и разные при случайном, как в V7); мёртвая точка шаблона М2 при θ = 0."""
import numpy as np
import torch

import models_v2.s16  # noqa: F401
from models_v2.s14 import v7_1m_fit as FT
from models_v2.s16 import x1_p0_dyn as DY, x1_p0_init as I, x1_p0_syn as SY


def test_router_data_gradient_is_exact_zero_at_zero_experts_and_nonzero_otherwise():
    b = DY.synthetic_batch()
    m = FT.make(I.SPEC6, 0)
    g = I.grads(m, lambda: I.split_loss(m, b, 0.1, 0.1)[0])
    assert I.group_norm(g, I.ROUTER) == 0.0 and I.group_norm(g, I.EXPERT) > 0
    with torch.no_grad():
        m.a += torch.tensor([0.3, -0.2], dtype=torch.float64)          # эксперты ненулевые
    g2 = I.grads(m, lambda: I.split_loss(m, b, 0.1, 0.1)[0])
    assert I.group_norm(g2, I.ROUTER) > 1e-6


def test_autograd_equals_central_difference():
    b = DY.synthetic_batch(n=120)
    m = FT.make(I.SPEC6, 1)
    with torch.no_grad():
        m.a += torch.tensor([0.1, 0.2], dtype=torch.float64)
    a = I.grads(m, lambda: I.split_loss(m, b, 1.0, 0.1)[2])
    n = I.numeric(m, lambda: I.split_loss(m, b, 1.0, 0.1)[2])
    assert max(float((a[k] - n[k]).abs().max()) for k in a) < 1e-7
    wrong = {k: v + (1e-3 if k == "a" else 0.0) for k, v in n.items()}      # мутация: подложенная ошибка градиента ловится тем же сравнением
    assert max(float((a[k] - wrong[k]).abs().max()) for k in a) > 1e-4


def test_experts_stay_copies_with_uniform_router_and_differ_with_random():
    b = DY.synthetic_batch()
    assert DY.symmetry_dist(b, router_zero=True, steps=60) == 0.0
    assert DY.symmetry_dist(b, router_zero=False, steps=60) > 1e-3


def test_synthetic_m2_theta_zero_is_dead_point_and_small_theta_is_not():
    d = SY.data()
    z = SY.grad_norms("m2", 0.0, d)
    assert z["gamma"] == 0.0 and z["theta"] == 0.0 and z["_F_equals_X1"] == 1.0
    s = SY.grad_norms("m2", 0.1, d)
    assert s["gamma"] > 0 and s["theta"] == 0.0                              # θ оживает только после γ ≠ 0
    assert SY.second_step_unblocks("m2", 0.1)["theta"] > 0
    for form in ("m1", "m3"):
        assert max(v for k, v in SY.grad_norms(form, 0.0, d).items() if not k.startswith("_") and k != "kappa") > 0
    assert np.isclose(SY.grad_norms("m1", 0.0, d)["kappa"], 0.0)            # затухание κ оживает только после a ≠ 0
