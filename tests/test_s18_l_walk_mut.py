"""S-L-IMPL-1 (`S_L_SPEC.md` разд. 6, мутации прохода): валидация на уровне обучения / со сдвинутым π, c и m контроля (а) по RV/W5-G, «закрыто + 1», перенос a в C. Каждая мутация роняет детектор; честный код проходит."""
import numpy as np

from models_v2.s18 import d_walk_syn as S
from models_v2.s18 import l_model as ML
from models_v2.s18 import l_walk as LW
from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_walk as W

P = np.arange(1000)
SH = np.array([0.0, 0.0, -0.3, -0.3, 0.0, 0.0])


def det_valid_true_pi(ctx, kind) -> list:
    """G месяца = QLIKE с настоящим π (numpy, независимо); пустой список = честно."""
    lv, c8, A = ctx.lv, ctx.x["C"].to_numpy(float), LW.arrays(ctx)
    _, _, G = LW.choose_l(ctx, P, kind)
    err = 0.0
    for bi, (T, Vb) in enumerate(W.blocks(P)):
        c, m = ML.stats_l(kind, A, T)
        for lam in W.LAMS:
            a = ML.fit_kind(kind, A, T, lam, c, m)
            f = lambda aa: (1 - lv.pi[Vb]) * lv.lnb[Vb] + lv.pi[Vb] * (np.log(c8[Vb]) + aa) + lv.bg[Vb]
            err = max(err, abs(float(RL.ell(ctx.rv[Vb], f(a)).mean() - RL.ell(ctx.rv[Vb], f(0.0)).mean()) - G[lam][bi]))
    return [] if err < 1e-8 else ["G ≠ QLIKE с настоящим π (%.2e)" % err]


def det_control_a(ctx) -> list:
    r = LW.fit_month_l(ctx, P, ctx.dp[1000:1020], ctx.x["C"].to_numpy(float)[1000:1020], "pibar")
    return [] if r["lam"] == 0.0 and r["a"] == 0.0 else ["rl1a: λ* или â ≠ 0"]


def test_honest_passes_detectors():
    ctx = S.syn_ctx("F", ths=SH)
    assert det_control_a(ctx) == [] and det_valid_true_pi(ctx, "shift252") == [] and det_valid_true_pi(ctx, "pibar") == []


def test_mutation_validation_with_shifted_pi_is_caught(monkeypatch):
    ctx = S.syn_ctx("F", ths=SH)
    monkeypatch.setattr(LW, "lnf_fr", lambda a, A, sel, lnf0: ML.ln_ftrain(a, A["lnb"][sel], ML.train_pi("shift252", A["pi"], sel), A["bg"][sel], A["lnc8"][sel]))
    assert det_valid_true_pi(ctx, "shift252")


def test_mutation_validation_with_mean_pi_is_caught_on_srl1(monkeypatch):
    ctx = S.syn_ctx("F", ths=SH)
    monkeypatch.setattr(LW, "lnf_fr", lambda a, A, sel, lnf0: ML.ln_ftrain(a, A["lnb"][sel], ML.train_pi("pibar", A["pi"], sel), A["bg"][sel], A["lnc8"][sel]))
    assert det_valid_true_pi(ctx, "pi")


def test_mutation_control_a_c_m_from_rv_over_w5g_is_caught(monkeypatch):
    ctx, orig = S.syn_ctx("F", ths=SH), ML.stats_l
    monkeypatch.setattr(ML, "stats_l", lambda kind, A, T: orig("pi", A, T))
    assert det_control_a(ctx)


def test_a_is_carried_into_C_as_cfg8_times_exp_a():
    ctx = S.syn_ctx("F", ths=SH)
    c0 = ctx.x["C"].to_numpy(float)[1000:1020]
    r = LW.fit_month_l(ctx, P, ctx.dp[1000:1020], c0, "pi")
    assert r["a"] != 0.0 and np.array_equal(r["C"], c0 * np.exp(r["a"])) and not np.array_equal(r["C"], c0)


def test_mutation_closed_plus_one_pool_changes_month_and_is_caught():
    ctx = S.syn_ctx("F", ths=SH)
    r0 = LW.fit_month_l(ctx, P, ctx.dp[1000:1020], ctx.x["C"].to_numpy(float)[1000:1020], "pi")
    r1 = LW.fit_month_l(ctx, np.arange(1001), ctx.dp[1000:1020], ctx.x["C"].to_numpy(float)[1000:1020], "pi")
    assert r0["a"] != r1["a"] or not np.array_equal(r0["G"], r1["G"]) or r0["n_train"] != r1["n_train"]
