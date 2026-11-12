"""S-RCD-IMPL-1 (Ж-259), тестировщик: `rd_model` — известные ответы на синтетике и вложение на реальных данных (M.G1–M.G4). Автор кода тесты не менял. Всё только читает; в сеть и на диск не пишет."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models_v2.s18 import cm_model as CM
from models_v2.s18 import r_loss as RL
from models_v2.s18 import rd_model as RM
from models_v2.s18 import rd_model_chk as MC
import test_s18_rd_model_h as H


def test_forms_lnc_of_known_answers():
    assert H.form_problems() == []


def test_bad_form_is_refused_everywhere():
    D, lv = H.syn(50)
    V = RL.view(D, lv, np.arange(50))
    for call in (lambda: RM.lnc_of(torch.zeros(6), V, "CX"), lambda: RM.starts("CX"), lambda: RM.bounds("CX"), lambda: RM.loss(torch.zeros(6), V, "CX", 0.0, 2.0, 1.0)):
        with pytest.raises(ValueError):
            call()


def test_starts_are_cm_x0_and_seed_zero_is_zero():
    for form in RM.FORMS:
        st = RM.starts(form)
        assert len(st) == 5 and not st[0].any() and all(np.array_equal(a, CM._x0(form, s)) for a, s in zip(st, RL.SEEDS))
        lo, hi = np.array(RM.bounds(form)).T
        assert all((a >= lo).all() and (a <= hi).all() for a in st)


def test_bounds_per_form_and_level_shift_is_recovered_with_cd_bounds():
    assert H.bounds_problems() == []


def test_plain_is_ordinary_qlike_without_clipping_known_answer():
    assert H.plain_problems() == []


def test_r_loss_equals_independent_numpy_rho_kappa():
    assert H.rho_problems() == []


@pytest.mark.parametrize("form", RM.FORMS)
def test_autograd_equals_finite_differences(form):
    assert H.grad_problems(form) == []


def test_loss_ck_equals_r_loss_bitwise_both_modes():
    D, lv = H.syn(500, 2)
    V = RL.view(D, lv, np.arange(500))
    c, m = RL.clip_stats(H.x0_of(D, lv, "CK"), np.arange(500))
    for s in range(4):
        v = torch.as_tensor(H.rand_theta("CK", s))
        for mode in RM.MODES:
            assert torch.equal(RM.loss(v, V, "CK", 1e-2, c, m, mode), RL.loss_r(v, V, "CK", 1e-2, c, m, mode))


@pytest.mark.parametrize("form", RM.FORMS)
def test_level_score_is_zero_at_theta_zero(form):
    D, lv = H.syn(500, 4)
    V = RL.view(D, lv, np.arange(500))
    c, m = RL.clip_stats(H.x0_of(D, lv, form), np.arange(500))
    s = torch.zeros((), dtype=torch.float64, requires_grad=True)
    RM.loss(torch.zeros(6, dtype=torch.float64), V, form, 0.0, c, m, "R", shift=s).backward()
    assert abs(float(s.grad)) <= 1e-12


def test_fit_ck_equals_fit_r_bitwise_and_spread_is_ptp():
    D, lv = H.syn(500, 8, th=np.array([0.2, 0.3, 0.1, 0.0, 0.2, 0.0]), form="CK")
    V = RL.view(D, lv, np.arange(500))
    c, m = RL.clip_stats(H.x0_of(D, lv, "CK"), np.arange(500))
    new, sp = RM.fit_full(V, "CK", 1e-2, c, m, "R")
    assert np.array_equal(new[0], RL.fit_r(V, "CK", 1e-2, c, m, "R")[0]) and np.array_equal(RM.fit(V, "CK", 1e-2, c, m, "R")[0], new[0])
    assert isinstance(sp, float) and sp >= 0.0 and len(new) == 1


def test_fit_cd_restarts_agree_convex():
    D, lv = H.syn(500, 9, th=np.array([-0.1, 0.4, 0.2, 0.2, 0.0, 0.1]))
    V = RL.view(D, lv, np.arange(500))
    c, m = RL.clip_stats(H.x0_of(D, lv, "CD"), np.arange(500))
    assert RM.fit_full(V, "CD", 1e-3, c, m, "R")[1] < 1e-6


def test_tie_between_restarts_gives_smaller_seed(monkeypatch):
    D, lv = H.syn(60, 1)
    V = RL.view(D, lv, np.arange(60))
    monkeypatch.setattr(RM, "minimize", lambda fun, x0, **kw: SimpleNamespace(fun=1.0, x=np.full(6, float(np.sum(x0)))))
    th, sp = RM.fit_full(V, "CD", 1e-2, 2.0, 1.0, "R")
    assert np.array_equal(th[0], np.full(6, float(np.sum(RM.starts("CD")[0])))) and sp == 0.0


def test_predict_lnc_is_mean_over_thetas():
    D, lv = H.syn(80, 1)
    V = RL.view(D, lv, np.arange(80))
    ths = [H.rand_theta("CD", 1), H.rand_theta("CD", 2)]
    want = np.mean([RM.lnc_of(torch.as_tensor(t), V, "CD").numpy() for t in ths], axis=0)
    assert np.array_equal(RM.predict_lnc(ths, V, "CD"), want) and np.array_equal(RM.predict_lnc(ths[:1], V, "CD"), RM.lnc_of(torch.as_tensor(ths[0]), V, "CD").numpy())


@pytest.mark.parametrize("mode", RM.MODES)
def test_synthetic_slope_on_zc_is_recovered(mode):
    est = []
    for s in range(10):     # медиана по 10 выборкам (спецификация: 20 переоценок)
        D, lv = H.syn(1200, 110 + s, th=np.array([0, 0, 0, 0.3, 0, 0.0]))
        V = RL.view(D, lv, np.arange(1200))
        c, m = RL.clip_stats(H.x0_of(D, lv, "CD"), np.arange(1200))
        est.append(RM.fit(V, "CD", 1e-3, c, m, mode)[0][3])
    assert abs(np.median(est) - 0.3) <= 0.1


def test_pit_c_m_and_loss_do_not_see_rows_outside_pool():
    assert H.pit_problems() == []


def test_real_nest_theta_zero_cd_ck_and_r_loss_on_2198_dates():
    D, x, lv = MC.load()
    V = RL.view(D, lv, np.arange(len(lv.pi)))
    z = torch.zeros(6, dtype=torch.float64)
    a, b, r = RM.lnc_of(z, V, "CD"), RM.lnc_of(z, V, "CK"), RL.lnc_of(z, V, "CK")
    assert len(a) == 2198 and torch.equal(a, b) and torch.equal(b, r)
    m1 = MC.months(x)[1]
    P = RL.pool_of("CK", D, x, m1)
    VP, (c, m) = RL.view(D, lv, P), RL.clip_stats(RL.x0_of(V, "CK"), P)
    v = torch.as_tensor(H.rand_theta("CK", 5))
    assert all(torch.equal(RM.loss(v, VP, "CK", 1e-2, c, m, md), RL.loss_r(v, VP, "CK", 1e-2, c, m, md)) for md in RM.MODES)
