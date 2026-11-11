"""S-A-IMPL-2 (тестировщик): вложение (θ = 0 ⇒ C_A = cfg8, H_A = K1), формулы 3.1–3.2 против независимого эталона, градиенты, штраф, старты, границы. Реальные K1 и cfg8 — только чтение."""
import numpy as np
import pytest
import torch

from models_v2.s18 import a_model as M
import s18_a_model_checks as C
from s18_a_guard import guard_real_files  # noqa: F401  (autouse)


@pytest.fixture(scope="module")
def D():
    return M.load_data()


def test_data_shape_2451_pairs_2198_with_cfg8(D):
    assert len(D.d) == 2451 and D.has8.sum() == 2198 and D.d[D.has8][0] == np.datetime64("2018-01-03")
    assert (D.tp > D.d).all()                                  # цель — строго следующий день


def test_zero_params_give_cfg8_and_k1_all_variants(D):
    C.check_nest(D)


def test_forward_equals_independent_reference_at_random_params(D):
    C.check_formula(D)


def test_gradients_at_zero_finite_nonzero_psi_zero_at_omega_zero(D):
    V = D.view(np.arange(len(D.d)))
    v = C.zero().clone().requires_grad_(True)
    M.loss(v, V, "A2", C.LAM).backward()
    g = v.grad.numpy()
    assert np.isfinite(g).all() and np.linalg.norm(g) > 0
    for sl in (M.I_D, M.I_W, M.I_C):
        assert np.linalg.norm(g[sl]) > 0
    assert np.linalg.norm(g[M.I_DW]) > 0 and np.linalg.norm(g[M.I_DK]) > 0
    assert np.abs(g[M.I_P]).max() == 0.0                       # ω = 0 ⇒ ψ градиента не получает (записано в спецификации)


def test_autograd_equals_numerical_gradient(D):
    C.check_grad(D)


def test_penalty_known_answer_per_block():
    C.check_penalty()


def test_starts_seed0_zero_others_vary_and_free_mask():
    C.check_starts()
    assert M.free_mask("A0", True).sum() == 7 and M.free_mask("A1", True).sum() == 16 and M.free_mask("A2", True).sum() == 28
    assert M.free_mask("A2p", True).sum() == 28 and M.free_mask("A2", False).sum() == 26
    assert not np.allclose(M.tau_of("A2"), M.tau_of("A2p")) and sorted(M.tau_of("A2p")) == sorted(M.tau_of("A2"))


def test_w_eff_clamped_to_unit_interval_known_answer(D):
    """δw = 5 не выводит w + δw за [min(0, w), max(1, w)]: вес якоря ограничен (разд. 3.2)."""
    V = D.view(np.arange(len(D.d)))
    x = np.zeros(M.NP)
    x[M.I_DW] = 5.0
    lnH, lnC = M.forward(torch.as_tensor(x), V, "A0")
    h8 = D.has8
    want = lnH.numpy() + np.maximum(D.w, 1.0) * (D.lnb - lnH.numpy()) + D.m
    assert np.abs(lnC.numpy()[h8] - want[h8]).max() <= 1e-12


def test_fit_respects_bounds_and_free_mask_on_real_short_window(D):
    V = D.view(np.arange(250, 900))
    for var in ("A1", "A2"):
        x = M.fit(V, var, 1e-3, seed=1, maxiter=60)
        assert x[M.I_W].min() >= 0 and x[M.I_W].max() <= 1 and np.abs(x[M.I_P]).max() <= 8 and abs(x[M.I_DK]) <= 1
        assert not x[~M.free_mask(var, True)].any()
    a0 = M.fit(V, "A0", 1e-3, seed=0, maxiter=60)
    assert not a0[M.I_W].any() and not a0[M.I_P].any() and not a0[M.I_C].any()
