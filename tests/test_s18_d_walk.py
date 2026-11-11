"""S-D-IMPL-1 (Ж-256; `S_D_SPEC.md` разд. 3, 6): абляции S-RCKm и S-RCKf — известные ответы на синтетике (уровень F ≠ уровню C) и умолчания = прежний S-RCK на боевых данных."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s18 import d_walk as D
from models_v2.s18 import d_walk_syn as S
from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_walk as W


def test_default_ctx_is_the_old_one_and_a_with_c_refused():
    ctx = S.syn_ctx("F")
    P = np.arange(900)
    assert ctx.x0_train is ctx.x0 and ctx.lv_train is None
    for k in ("l_pi", "l_lnb", "l_bg", "rv", "c8"):
        assert np.array_equal(getattr(ctx.view_train(P), k).numpy(), getattr(ctx.view(P), k).numpy())
    with pytest.raises(ValueError):
        W.make_ctx("A", "C")
    with pytest.raises(ValueError):
        W.make_ctx("CK", "X")


def test_defaults_reproduce_saved_rck_and_rckb_month_bitwise():
    assert S.saved_month_problems() == []


def test_rckm_known_answer_m_is_one_c_is_quantile_of_rv_over_w5g():
    assert S.problems("rckm", S.syn_ctx("F")) == []


def test_rckf_known_answer_c_m_from_rv_over_cfg8_validation_on_level_f():
    assert S.problems("rckf", S.syn_ctx("C")) == []


def test_m_one_really_switches_off_the_correction():
    ctx, P = S.syn_ctx("F"), np.arange(1000)
    c8 = ctx.x["C"].to_numpy(float)[1000:1020]
    r1, r0 = (W.fit_month(ctx, P, ctx.dp[1000:1020], c8, mode) for mode in ("Rm1", "R"))
    assert r1["m"] == 1.0 and r0["m"] < 0.99 and r1["c"] == r0["c"] and np.isfinite(r1["c"])
    assert 0.0 < r0["m"] < 1.0 and r1["c"] > 1.0


def test_level_score_at_theta_zero_is_zero_on_level_c_and_nonzero_on_level_f():
    f, c = S.syn_ctx("F"), S.syn_ctx("C")
    P = np.arange(1000)
    sf = S.level_score(RL.view(f.D, f.lv, P), *RL.clip_stats(f.x0, P))
    sc = S.level_score(c.view_train(P), *RL.clip_stats(c.x0_train, P))
    assert abs(sc) <= 1e-12 < 1e-6 < abs(sf), (sc, sf)


def test_level_c_ctx_trains_on_pi_one_but_validates_on_level_f():
    f, c = S.syn_ctx("F"), S.syn_ctx("C")
    P = np.arange(500)
    v = c.view_train(P)
    assert (v.l_pi == 1).all() and (v.l_lnb == 0).all() and (v.l_bg == 0).all()
    assert np.array_equal(c.view(P).l_pi.numpy(), f.view(P).l_pi.numpy()) and np.array_equal(c.lnf0, f.lnf0)
    assert np.allclose(c.x0_train, c.rv / c.x["C"].to_numpy(float), rtol=1e-12, atol=0) and not np.allclose(c.x0_train, f.x0)


def test_assemble_rckf_takes_architecture_from_ctx_not_from_vars():
    assert "rckf" not in W.VARS and S.assemble_problems() == []
    assert D.VARS_D["rckf"] == ("CK", "R", "blocks") and D.VARS_D["rckm"] == ("CK", "Rm1", "blocks") and D.LEVEL_D == {"rckm": "F", "rckf": "C"}
