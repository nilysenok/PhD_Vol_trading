"""S-D-IMPL-1: обязательные мутации абляций S-RCKm / S-RCKf (`S_D_SPEC.md` разд. 6; mutations.md № 30, 144, 164) — каждая ловится детектором `d_walk_syn` или D1.G1."""
import numpy as np

from models_v2.s16 import x1_ya_walk as YA
from models_v2.s18 import d_walk_syn as S
from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_walk as W
from models_v2.s18 import r_walk_syn as RS

ORIG_MAKE, ORIG_STATS = W.make_ctx, W.stats_of


def test_honest_detectors_are_clean():
    assert S.problems("rckm", S.syn_ctx("F")) == [] and S.problems("rckf", S.syn_ctx("C")) == [] and S.assemble_problems() == []


def test_mut_default_train_level_c_is_caught(monkeypatch):
    monkeypatch.setattr(W, "make_ctx", lambda arch, train_level="C": ORIG_MAKE(arch, train_level))
    assert S.saved_month_problems() != []


def test_mut_rm1_keeps_m_is_caught(monkeypatch):
    monkeypatch.setattr(W, "stats_of", lambda x0, T, mode: RL.clip_stats(x0, T))
    assert S.problems("rckm", S.syn_ctx("F")) != []


def test_mut_rm1_without_clip_is_caught(monkeypatch):
    monkeypatch.setattr(W, "stats_of", lambda x0, T, mode: (float("inf"), 1.0))
    monkeypatch.setitem(W.LOSS_MODE, "Rm1", "plain")
    assert S.problems("rckm", S.syn_ctx("F")) != []


def test_mut_rckf_validation_on_level_c_is_caught(monkeypatch):
    monkeypatch.setattr(W.Ctx, "view", W.Ctx.view_train)
    assert S.problems("rckf", S.syn_ctx("C")) != []


def test_mut_rckf_base_of_g_on_level_c_is_caught():
    ctx = S.syn_ctx("C")
    ctx.lnf0 = np.log(ctx.x["C"].to_numpy(float))
    assert S.problems("rckf", ctx) != []


def test_mut_rckf_c_m_from_rv_over_w5g_is_caught():
    ctx = S.syn_ctx("C")
    ctx.x0_train = ctx.x0
    assert S.problems("rckf", ctx) != []


def test_mut_rckf_loss_on_level_f_is_caught():
    assert S.problems("rckf", S.syn_ctx("C", train=False)) != []


def test_mut_closed_plus_one_pool_is_caught(monkeypatch):
    monkeypatch.setattr(YA, "closed_date", lambda tp, d: d)
    assert RS.pool_problems() != []


def test_mut_assemble_by_vars_is_caught(monkeypatch):
    orig = W.assemble

    def mut(var, ctx, parts):
        W.VARS[var]    # прежний код: архитектура из VARS[var]
        return orig(var, ctx, parts)
    monkeypatch.setattr(W, "assemble", mut)
    assert S.assemble_problems() != []


def test_level_score_nonzero_when_rckf_trains_on_level_f():
    f = S.syn_ctx("F")
    P = np.arange(1000)
    assert abs(S.level_score(RL.view(f.D, f.lv, P), *RL.clip_stats(f.x0, P))) > 1e-6
