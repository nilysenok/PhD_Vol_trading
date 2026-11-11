"""S-A-IMPL-2, мутации листа (S_A_SPEC разд. 8): семь подложенных ошибок обязаны ронять проверки (№ 103, 144, 164-образцы: через боевой путь). Только monkeypatch (№ 43)."""
import copy

import numpy as np
import pytest
import torch

from models_v2.s18 import a_feat as A
from models_v2.s18 import a_model as M
import s18_a_model_checks as C
from s18_a_guard import guard_real_files  # noqa: F401  (autouse)


@pytest.fixture(scope="module")
def D():
    return M.load_data()


def test_honest_passes_all_checks(D):
    C.check_nest(D), C.check_formula(D, seeds=(1,)), C.check_penalty(), C.check_grad(D, n_points=1), C.check_starts()


def test_mutation_penalty_without_dw_dk_caught(monkeypatch):
    def pen(v, lam):
        return lam * ((v[M.I_D] ** 2).sum() + (v[M.I_W] ** 2).sum() + (v[M.I_C] ** 2).sum()) + M.PSI_PEN * (v[M.I_P] ** 2).sum()
    monkeypatch.setattr(M, "penalty", pen)
    with pytest.raises(AssertionError):
        C.check_penalty()


def test_mutation_m_taken_as_G_anchor_counted_twice_caught(D):
    G = copy.copy(D)
    ell = D.lnb - D.lnk1                              # на нуле H_A = K1
    G.m = np.where(D.has8, D.m + D.w * ell, 0.0)      # m = G вместо G − w·ℓ (ошибка W9 Д-1)
    with pytest.raises(AssertionError):
        C.check_nest(G)


def test_mutation_ell_by_k1_instead_of_h_a_caught(D, monkeypatch):
    def fwd(v, V, var, corr=True):
        lnH, _ = M.forward(v, V, var, corr)
        w = torch.minimum(torch.maximum(V.w + v[M.I_DW], torch.clamp(V.w, max=0.0)), torch.clamp(V.w, min=1.0))
        lnC = lnH + w * (V.lnb - V.lnk1) + (1.0 + v[M.I_DK]) * V.m          # ℓ по K1
        return lnH, torch.where(V.has8, lnC, lnH)
    C.check_formula(D, seeds=(1,))
    with pytest.raises(AssertionError):
        C.check_formula(D, fwd, seeds=(1,))


def test_mutation_pairs_without_cfg8_with_g_nonzero_caught(D):
    def fwd(v, V, var, corr=True):
        lnH, _ = M.forward(v, V, var, corr)
        w = torch.minimum(torch.maximum(V.w + v[M.I_DW], torch.clamp(V.w, max=0.0)), torch.clamp(V.w, min=1.0))
        return lnH, lnH + w * (V.lnb - lnH) + (1.0 + v[M.I_DK]) * V.m     # без has8: g ≠ 0 у пар 2017
    with pytest.raises(AssertionError):
        C.check_formula(D, fwd, seeds=(1,))


def test_mutation_start_not_at_zero_uniform_kernel_caught(monkeypatch):
    orig = M._x0

    def x0(var, seed, free):
        x = orig(var, seed, free)
        return np.where(free, x, 0.0) + np.where(free & (np.arange(M.NP) < 8) & (np.arange(M.NP) >= 5), 1.0, 0.0)   # ω = 1: равномерное ядро G
    monkeypatch.setattr(M, "_x0", x0)
    with pytest.raises(AssertionError):
        C.check_starts()


def test_mutation_detach_in_s_caught_by_gradient_check(D, monkeypatch):
    def ln_s(F, theta, lag, tau=A.TAU):
        rb = A.rv_b(F, A.slot_exp(theta["c"], tau)).detach()                   # класс № 144
        cols = []
        for k, (ix, ok) in enumerate(lag):
            a = A.lag_weights(ok, k, theta["omega"][k], theta["psi"][k])
            cols.append(torch.log((a * rb[torch.as_tensor(ix), k]).sum(1)))
        return torch.stack(cols, dim=1)
    monkeypatch.setattr(A, "ln_S", ln_s)
    with pytest.raises(AssertionError):
        C.check_grad(D, n_points=1)


def test_mutation_seeds_do_not_change_start_caught(monkeypatch):
    orig = M._x0
    monkeypatch.setattr(M, "_x0", lambda var, seed, free: orig(var, 1 if seed else 0, free))
    with pytest.raises(AssertionError):
        C.check_starts()
    monkeypatch.setattr(M, "_x0", lambda var, seed, free: np.zeros(M.NP))
    with pytest.raises(AssertionError):
        C.check_starts()
