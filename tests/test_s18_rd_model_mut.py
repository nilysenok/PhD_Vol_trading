"""S-RCD-IMPL-1 (Ж-259), тестировщик: мутации `rd_model` (спецификация разд. 8.2; mutations.md № 144, 164). Каждая мутация подложена monkeypatch; детектор из `test_s18_rd_model_h` обязан быть непустым. На честном коде — пуст."""
import math
from types import SimpleNamespace

import numpy as np
import torch

from models_v2.s18 import cm_model as CM
from models_v2.s18 import r_loss as RL
from models_v2.s18 import rd_model as RM
import test_s18_rd_model_h as H

ORIG_LOSS, ORIG_VIEW = RM.loss, RL.view


def test_honest_detectors_are_clean():
    assert H.form_problems() == [] and H.bounds_problems() == [] and H.plain_problems() == [] and H.rho_problems() == [] and H.pit_problems() == []
    assert H.grad_problems("CD") == [] and H.grad_problems("CK") == []


def test_mut_ck_form_instead_of_cd_in_lnc_of_is_caught(monkeypatch):
    monkeypatch.setattr(RM, "lnc_of", lambda v, V, form: torch.log(V.c8 * CM.forward(v, V, "CK")["M"]))
    assert H.form_problems() != [] and H.rho_problems() != []


def test_mut_cd_form_instead_of_ck_in_lnc_of_is_caught(monkeypatch):
    monkeypatch.setattr(RM, "lnc_of", lambda v, V, form: torch.log(V.c8 * CM.forward(v, V, "CD")["M"]))
    assert H.form_problems() != []


def test_mut_eta_inside_other_logarithm_is_caught_bitwise(monkeypatch):
    monkeypatch.setattr(RM, "lnc_of", lambda v, V, form: torch.log(V.c8) + torch.log(CM.forward(v, V, form)["M"]))
    assert H.form_problems() != []


def test_mut_ck_bounds_for_cd_is_caught(monkeypatch):
    monkeypatch.setattr(RM, "bounds", lambda form: CM.bounds("CK"))
    assert H.bounds_problems() != []


def test_mut_theta0_bounds_zero_one_for_cd_is_caught(monkeypatch):
    monkeypatch.setattr(RM, "bounds", lambda form: [(0.0, 1.0)] + CM.bounds(form)[1:])
    assert H.bounds_problems() != []


def test_mut_detach_in_branch_u_above_ln_kappa_is_caught_by_gradient_check(monkeypatch):
    def rho(u, kappa):
        lk = math.log(kappa)
        tail = (kappa - 1.0 - lk) + (kappa - 1.0) * (u.detach() - lk)
        return torch.where(u <= lk, torch.exp(torch.clamp(u, max=lk)) - u - 1.0, tail)
    monkeypatch.setattr(RL, "rho", rho)
    assert H.grad_problems("CD") != [] and H.grad_problems("CK") != []


def test_mut_detach_of_whole_data_term_is_caught_by_gradient_check(monkeypatch):
    monkeypatch.setattr(RM, "loss", lambda v, V, form, lam, c, m, mode="R", shift=0.0: ORIG_LOSS(v.detach(), V, form, 0.0, c, m, mode, shift) + CM.penalty(v, lam))
    assert H.grad_problems("CD") != []


def test_mut_plain_with_clipping_is_caught(monkeypatch):
    monkeypatch.setattr(RM, "loss", lambda v, V, form, lam, c, m, mode="R", shift=0.0: ORIG_LOSS(v, V, form, lam, c, m, "R", shift))
    assert H.plain_problems() != []


def test_mut_plain_with_m_calibration_is_caught(monkeypatch):
    monkeypatch.setattr(RM, "loss", lambda v, V, form, lam, c, m, mode="R", shift=0.0: ORIG_LOSS(v, V, form, lam, c, m, mode, shift + (math.log(m) if mode == "plain" else 0.0)))
    assert H.plain_problems() != []


def test_mut_pi_set_to_one_silently_level_c_is_caught(monkeypatch):
    monkeypatch.setattr(RL, "ln_f", lambda lnc, V: lnc)
    assert H.rho_problems() != [] and H.plain_problems() != []


def test_mut_pi_of_next_day_t_plus_instead_of_d_is_caught(monkeypatch):
    def view(D, lv, sel):
        V = ORIG_VIEW(D, lv, sel)
        V.l_pi = torch.as_tensor(np.roll(np.asarray(lv.pi, float), -1)[V.sel])
        return V
    monkeypatch.setattr(RL, "view", view)
    assert H.rho_problems() != [] and H.pit_problems() != []


def test_mut_clip_stats_over_all_rows_is_caught_by_pit(monkeypatch):
    orig = RL.clip_stats
    monkeypatch.setattr(RL, "clip_stats", lambda x0, T: orig(x0, np.arange(len(x0))))
    assert H.pit_problems() != []


def test_mut_penalty_only_on_part_of_parameters_is_caught(monkeypatch):
    monkeypatch.setattr(CM, "penalty", lambda v, lam: lam * (v[1:] ** 2).sum())
    assert H.plain_problems() != [] and H.rho_problems() != []


def test_mut_m_ignored_in_r_loss_breaks_level_score(monkeypatch):
    def loss(v, V, form, lam, c, m, mode="R", shift=0.0):
        return ORIG_LOSS(v, V, form, lam, c, 1.0, mode, shift)
    monkeypatch.setattr(RM, "loss", loss)
    D, lv = H.syn(500, 4)
    V = RL.view(D, lv, np.arange(500))
    c, m = RL.clip_stats(H.x0_of(D, lv, "CD"), np.arange(500))
    s = torch.zeros((), dtype=torch.float64, requires_grad=True)
    RM.loss(torch.zeros(6, dtype=torch.float64), V, "CD", 0.0, c, m, "R", shift=s).backward()
    assert abs(float(s.grad)) > 1e-3 and H.rho_problems() != []


def test_mut_tie_between_restarts_takes_larger_seed_is_caught(monkeypatch):
    class Np:
        def __getattr__(self, k):
            return getattr(np, k)

        @staticmethod
        def argmin(a):
            a = np.asarray(a)
            return int(len(a) - 1 - np.argmin(a[::-1]))
    monkeypatch.setattr(RM, "np", Np())
    monkeypatch.setattr(RM, "minimize", lambda fun, x0, **kw: SimpleNamespace(fun=1.0, x=np.full(6, float(np.sum(x0)))))
    D, lv = H.syn(60, 1)
    th = RM.fit(RL.view(D, lv, np.arange(60)), "CD", 1e-2, 2.0, 1.0, "R")[0]
    assert not np.array_equal(th, np.full(6, float(np.sum(RM.starts("CD")[0]))))     # честный код вернул бы seed 0
