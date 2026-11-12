"""S-RCD-IMPL-2 (Ж-259), тестировщик: мутации прохода `rd_walk` (спецификация разд. 8.3; mutations.md № 144, 164). Каждая мутация подложена через `patched`/`monkeypatch` на боевой путь `RD.month` / `choose_*`; детектор обязан быть непустым."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s16 import x1_ya_walk as YA
from models_v2.s18 import cm_model as CM
from models_v2.s18 import cm_walk as CW
from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_walk as W
from models_v2.s18 import rd_walk as RD
import test_s18_rd_walk_h as K
import test_s18_rd_walk_h2 as K2

ORIG_DECIDE, ORIG_BLOCKS, ORIG_ELL, ORIG_VIEW, ORIG_STATS = W.decide, W.blocks, RL.ell, RD._view, W.stats_of
ORIG_R, ORIG_SC = RD.fit_month_r, RD.fit_month_sc


def test_honest_detectors_are_clean():
    assert K.spy_problems() == [] and K.rule_problems() == [] and K.zero_problems() == [] and K.incl_problems() == [] and K.sc_rule_problems() == []
    assert K2.args_problems() == [] and K2.level_val_problems() == [] and K2.level_train_problems() == [] and K2.pit_problems() == []


def test_mut_closed_plus_one_closed_date_is_d_is_caught(monkeypatch):     # № 164
    monkeypatch.setattr(YA, "closed_date", lambda tp, d: d)
    assert K.spy_problems() != [] and K2.pit_problems() != []


def test_mut_theta_zero_absent_from_rcd_grid_is_caught(monkeypatch):
    def mut(G, rule):
        w = {lam: sum(g < 0 for g in G[lam]) for lam in W.LAMS}
        return max(W.LAMS, key=lambda lam: (w[lam], lam)), w
    monkeypatch.setattr(W, "decide", mut)
    assert K.rule_problems() != [] and K.zero_problems() != []


def test_mut_rule_one_of_three_is_caught(monkeypatch):
    monkeypatch.setattr(W, "MIN_W", 1)
    assert K.rule_problems() != []


def test_mut_tie_takes_smaller_lambda_in_rcd_is_caught(monkeypatch):
    def mut(G, rule):
        lam, w = ORIG_DECIDE(G, rule)
        ok = [k for k in W.LAMS if w[k] >= W.MIN_W]
        return (min(k for k in ok if w[k] == max(w[j] for j in ok)) if ok else lam), w
    monkeypatch.setattr(W, "decide", mut)
    assert K.rule_problems() != []


def test_mut_training_block_includes_validation_block_is_caught(monkeypatch):     # № 144
    monkeypatch.setattr(W, "blocks", lambda P, nb=W.NB: [(np.asarray(P)[: len(P) - W.VAL * (nb - b - 1)], V) for b, (T, V) in enumerate(ORIG_BLOCKS(P, nb))])
    assert K.rule_problems() != []


def test_mut_validation_in_rho_kappa_instead_of_ordinary_qlike_is_caught(monkeypatch):
    monkeypatch.setattr(RL, "ell", lambda rv, lnf: np.minimum(ORIG_ELL(rv, lnf), 0.3))
    assert K.rule_problems() != [] and K.sc_rule_problems() != []


def test_mut_r_in_control_clip_and_m_in_scdf_is_caught():
    assert K2.args_problems(alter=lambda form, lam, c, m, mode: (form, lam, 2.0, 0.9, "R")) != []


@pytest.mark.parametrize("alter", [lambda f, l, c, m, md: (f, l, c, 0.9, md), lambda f, l, c, m, md: (f, l, 5.0, m, md), lambda f, l, c, m, md: (f, l, c, m, "R")], ids=["m_not_one", "clip_c_finite", "mode_R"])
def test_mut_scdf_control_arguments_are_caught_one_by_one(alter):
    assert K2.args_problems(alter=lambda f, l, c, m, md: alter(f, l, c, m, md) if md == "plain" else (f, l, c, m, md)) != []


def test_mut_scdf_runs_rcd_protocol_blocks_and_theta_zero_is_caught(monkeypatch):
    monkeypatch.setitem(RD.VARS, "cdf", ("CD", "R"))
    assert K2.args_problems() != []


def test_mut_theta_zero_in_scdf_lambda_grid_is_caught(monkeypatch):
    monkeypatch.setattr(CM, "LAMS", (0.0,) + CM.LAMS)
    assert K.sc_rule_problems() != []


def test_mut_sc_tie_takes_smaller_lambda_is_caught(monkeypatch):
    monkeypatch.setattr(CM, "LAMS", CM.LAMS[::-1])
    assert K.sc_rule_problems() != []


def test_mut_sc_validation_window_not_126_is_caught(monkeypatch):
    monkeypatch.setattr(CW, "VAL", 100)
    assert K.sc_rule_problems() != []


def test_mut_sc_short_pool_still_validates_is_caught(monkeypatch):
    monkeypatch.setattr(CW, "MIN_VALID", 100)
    assert K.sc_rule_problems() != [] and K.incl_problems() != []


def test_mut_scdf_validation_at_level_c_is_caught(monkeypatch):
    monkeypatch.setattr(RD, "_view", lambda ctx, sel, level: ctx.view_train(sel) if len(sel) == CW.VAL else ORIG_VIEW(ctx, sel, level))
    assert K2.level_val_problems() != []


def test_mut_ckf_trained_at_level_c_copy_of_ck_is_caught(monkeypatch):
    monkeypatch.setattr(RD, "_view", lambda ctx, sel, level: ctx.view_train(sel))
    assert K2.level_train_problems() != [] and K2.level_val_problems() != []


def test_mut_scdf_included_from_2020_08_instead_of_2018_07_is_caught(monkeypatch):
    monkeypatch.setattr(YA, "ACTIVATION", pd.Timestamp("2020-08-03"))
    assert K.incl_problems() != []


def test_mut_rcd_included_before_630_pairs_is_caught(monkeypatch):
    monkeypatch.setattr(W, "MIN_POOL", 120)
    assert K.incl_problems() != []


@pytest.mark.parametrize("swap", [{"rcd": ("CK", "R")}, {"cdf": ("CK", "SC")}, {"ckf": ("CD", "SC")}], ids=["rcd_ck", "cdf_ck", "ckf_cd"])
def test_mut_form_ck_instead_of_cd_and_back_in_pass_is_caught(monkeypatch, swap):
    for k, v in swap.items():
        monkeypatch.setitem(RD.VARS, k, v)
    assert K.spy_problems() != [] and K2.args_problems() != []


def test_mut_features_of_day_d_plus_1_in_forecast_is_caught(monkeypatch):
    monkeypatch.setattr(RD, "fit_month_r", lambda ctx, form, P, days, c0: ORIG_R(ctx, form, P, np.asarray(days) + 1, c0))
    monkeypatch.setattr(RD, "fit_month_sc", lambda ctx, form, P, days, c0, level="F": ORIG_SC(ctx, form, P, np.asarray(days) + 1, c0, level))
    assert K2.pit_problems() != []


def test_mut_clip_stats_over_all_rows_rcd_is_caught(monkeypatch):
    monkeypatch.setattr(W, "stats_of", lambda x0, T, mode: ORIG_STATS(x0, np.arange(len(x0)), mode))
    assert K2.pit_problems() != []


def test_mut_scdf_validation_sees_future_rows_is_caught(monkeypatch):
    monkeypatch.setattr(RD, "_view", lambda ctx, sel, level: ORIG_VIEW(ctx, np.concatenate([sel, np.arange(sel.max() + 1, sel.max() + 51)]) if len(sel) == CW.VAL else sel, level))
    assert K2.pit_problems() != []
