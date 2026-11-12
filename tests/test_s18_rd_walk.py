"""S-RCD-IMPL-2 (Ж-259), тестировщик: проход `rd_walk` — известные ответы на синтетике (правило выбора, блоки, пул, θ = 0 в сетке, включение, протокол SC, уровень F, PIT месяца). Мутации — `test_s18_rd_walk_mut`."""
import numpy as np
import pandas as pd
import torch

from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_walk as W
from models_v2.s18 import rd_walk as RD
import test_s18_rd_walk_h as K
import test_s18_rd_walk_h2 as K2


def test_api_vars_ids_outs():
    assert RD.VARS == {"rcd": ("CD", "R"), "cdf": ("CD", "SC"), "ckf": ("CK", "SC")} and RD.IDS == {"rcd": "S-RCD", "cdf": "S-CDF", "ckf": "S-CKF"} and RD.OUTS == {"rcd": "rcd.parquet", "cdf": "cdf.parquet", "ckf": "ckf.parquet"}


def test_pool_is_t_plus_le_d_and_form_follows_the_series():
    assert K.spy_problems() == []


def test_rule_two_of_three_tie_larger_lambda_theta_zero_blocks_and_ordinary_qlike():
    assert K.rule_problems() == []


def test_zero_signal_selects_theta_zero_and_keeps_cfg8_bitwise():
    assert K.zero_problems() == []


def test_inclusion_630_for_rcd_and_2018_07_02_for_controls():
    assert K.incl_problems() == []


def test_sc_lambda_is_argmin_of_level_f_validation_tie_larger_and_short_pool_default():
    assert K.sc_rule_problems() == []


def test_controls_train_plain_without_clipping_and_c_is_cfg8_times_form():
    assert K2.args_problems() == []


def test_validation_is_level_f_not_level_c_known_answer_pi_zero():
    assert K2.level_val_problems() == []


def test_ckf_is_level_f_control_not_a_copy_of_ck():
    assert K2.level_train_problems() == []


def test_pit_month_ignores_future_rv_and_features():
    assert K2.pit_problems() == []


def test_strong_signal_rcd_selects_theta_and_improves_out_of_sample():
    ctx = K.ctx_syn(1800, 11, th=K.STRONG, sig=0.3, jump=0.0)
    days = np.arange(1500, 1800)
    r = RD.fit_month_r(ctx, "CD", np.arange(1500), days, ctx.x["C"].to_numpy(float)[days])
    V = ctx.view(days)
    lnf = RL.ln_f(torch.as_tensor(np.log(r["C"])), V).numpy()
    assert r["active"] and r["lam"] > 0 and np.abs(r["theta"]).max() > 0.05 and RL.ell(ctx.rv[days], lnf).mean() < RL.ell(ctx.rv[days], ctx.lnf0[days]).mean()


def test_zero_month_sc_and_inactive_rcd_month_are_cfg8_copies():
    c0 = np.arange(1.0, 6.0)
    z = RD.zero_month_sc(np.arange(7), c0)
    assert not z["active"] and np.array_equal(z["C"], c0) and z["C"] is not c0 and z["c"] == K.INF and z["m"] == 1.0 and z["n_train"] == 7 and (z["theta"] == 0).all()
    ctx = K.ctx_syn(900, 5)
    r = RD.fit_month_r(ctx, "CD", np.arange(100), np.arange(100, 105), ctx.x["C"].to_numpy(float)[100:105])
    assert not r["active"] and np.isnan(r["lam"]) and (r["theta"] == 0).all() and r["n_train"] == 100


def test_walk_positions_work_dir_and_skip_of_existing_month(tmp_path):
    ctx = K.ctx_syn(900, 5)
    a, b = K.month_at(ctx, "2018-07-02")
    out = RD.walk("cdf", ctx, positions={a}, work=tmp_path)
    assert [(x, y) for x, y, _ in out] == [(a, b)] and (tmp_path / ("m%03d.pkl" % a)).exists()
    again = RD.walk("cdf", ctx, positions={a}, work=tmp_path)
    assert again[0][2]["lam"] == out[0][2]["lam"] and np.array_equal(again[0][2]["C"], out[0][2]["C"])
    assert pd.DatetimeIndex(ctx.x["d"])[a] == pd.Timestamp("2018-07-02") and W.month_bounds(ctx)[0][0] == 0
