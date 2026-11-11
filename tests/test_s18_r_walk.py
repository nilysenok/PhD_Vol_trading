"""S-R-IMPL-2: проход R — известные ответы на синтетике (правило выбора, блоки, пул, θ = 0 в сетке, валидация, включение) и вложение на реальных данных (W.G2). Автор тестов — автор кода (по поручению)."""
import numpy as np
import pandas as pd
import pytest
import torch

from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_reg as RG
from models_v2.s18 import r_walk as W
from models_v2.s18 import r_walk_chk as K
from models_v2.s18 import r_walk_syn as S


def test_rule_blocks_and_single_known_answers():
    assert S.rule_problems() == []


def test_blocks_known_positions_and_t_before_v():
    assert S.block_problems() == []


def test_closed_pool_is_t_plus_le_d_for_both_architectures():
    assert S.pool_problems() == []


def test_zero_signal_picks_theta_zero_and_keeps_cfg8_bitwise():
    assert S.zero_problems() == []


def test_validation_is_plain_qlike_with_outlier_in_block():
    assert S.val_problems() == []


def test_inclusion_at_630_pairs_is_the_same_for_r_rb_rc():
    assert S.incl_problems() == []


def test_strong_signal_selects_theta_and_improves_out_of_sample():
    ths = np.array([0.0, 0.5, 0.0, 0.0, -0.6, 0.6])
    ctx = S.syn_ctx(1800, 11, ths=ths)
    days = np.arange(1500, 1800)
    r = W.fit_month(ctx, np.arange(1500), days, ctx.x["C"].to_numpy(float)[days], "R")
    assert r["active"] and r["lam"] > 0 and np.abs(r["theta"]).max() > 0.05
    V = ctx.view(days)
    lnf = RL.ln_f(torch.as_tensor(np.log(r["C"])), V).numpy()
    assert RL.ell(ctx.rv[days], lnf).mean() < RL.ell(ctx.rv[days], ctx.lnf0[days]).mean()


def test_inactive_month_returns_cfg8_bitwise_and_nan_lambda():
    ctx = S.syn_ctx(900, 2)
    c0 = ctx.x["C"].to_numpy(float)[700:710]
    r = W.fit_month(ctx, np.arange(500), np.arange(700, 710), c0, "R")
    assert not r["active"] and np.isnan(r["lam"]) and np.array_equal(r["C"], c0) and (r["theta"] == 0).all()


def test_nest_theta_zero_everywhere_equals_w5g_and_x1():
    assert K.check_nest() == []


@pytest.mark.parametrize("arch", ["A", "CK"])
def test_layers_are_over_c_r_not_cfg8(arch):
    assert S.layer_problems(K.ctx_of(arch)) == []


def test_assemble_columns_and_frozen_level_at_theta_zero():
    ctx = K.ctx_of("CK")
    df = K.zero_walk(ctx)
    for c in ("d", "t_plus", "rv", "cfg8", "C", "F_fr", "X1_R", "F", "pi", "beta_gap", "refit_date", "active", "lam", "w_1e-1", "w_1e-2", "w_1e-3", "G_1e-1_b1", "G_1e-3_b3", "c", "m", "n_train", "gam_c", "th_p1"):
        assert c in df.columns
    w5 = pd.read_parquet(RL.W5G)
    assert float(np.max(np.abs(df["F_fr"] / w5["F"] - 1.0))) < 1e-12 and not df["active"].any()


def test_unregistered_id_is_refused():
    with pytest.raises(SystemExit):
        RG.require("S-R-NET-TAKOGO")
