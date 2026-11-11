"""S-L-IMPL-1 (`S_L_SPEC.md` разд. 2, 3, 6): помесячный проход `l_walk` на синтетике (уровень F ≠ уровню C) — контроль (а) ≡ W5-G, сдвиг 0 у (б) = S-RL1, независимый пересчёт â и G, PIT пула."""
import copy

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from models_v2.s16 import x1_w3_core as C
from models_v2.s18 import d_walk_syn as S
from models_v2.s18 import l_model as ML
from models_v2.s18 import l_walk as LW
from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_walk as W

P = np.arange(1000)
SH = np.array([0.0, 0.0, -0.3, -0.3, 0.0, 0.0])    # синтетика: истинный сдвиг уровня C — один параметр −0,3
KEYS = ("lam", "w", "G", "c", "m", "a", "C", "n_train", "active")


def month(ctx, kind, pool=P):
    return LW.fit_month_l(ctx, pool, ctx.dp[1000:1020], ctx.x["C"].to_numpy(float)[1000:1020], kind)


def same(a, b, keys=KEYS) -> bool:
    return all(np.array_equal(np.asarray(a[k]), np.asarray(b[k]), equal_nan=True) for k in keys)


def np_loss(a, ctx, T, kind, c, m, lam):
    """Независимая потеря: numpy, без torch и без l_model."""
    lv, c8 = ctx.lv, ctx.x["C"].to_numpy(float)
    pi = {"pi": lv.pi[T], "pibar": np.full(len(T), lv.pi[T].mean()), "shift252": np.roll(lv.pi, 252)[T]}[kind]
    u = np.log(ctx.rv[T]) - ((1 - pi) * lv.lnb[T] + pi * (np.log(c8[T]) + a) + lv.bg[T]) - np.log(m)
    k, lk = c / m, np.log(c / m)
    r = np.where(u <= lk, np.exp(np.minimum(u, lk)) - u - 1.0, (k - 1.0 - lk) + (k - 1.0) * (u - lk))
    return r.mean() + lam * a * a


def test_control_a_month_is_w5g_exactly_lambda_zero_a_zero_g_zero():
    ctx = S.syn_ctx("F", ths=SH)
    r = month(ctx, "pibar")
    assert r["active"] and r["lam"] == 0.0 and r["a"] == 0.0 and np.array_equal(r["C"], ctx.x["C"].to_numpy(float)[1000:1020]) and (r["G"] == 0.0).all()


def test_control_b_with_shift_zero_equals_srl1_bitwise(monkeypatch):
    ctx = S.syn_ctx("F", ths=SH)
    r1 = month(ctx, "pi")
    monkeypatch.setattr(ML, "SHIFT", 0)
    assert same(r1, month(ctx, "shift252"))
    monkeypatch.setattr(ML, "SHIFT", 252)
    assert not same(r1, month(ctx, "shift252"), ("G",))


def test_srl1_shifts_level_down_and_matches_independent_numpy_fit():
    ctx = S.syn_ctx("F", ths=SH)
    r = month(ctx, "pi")
    assert r["lam"] > 0.0 and r["a"] < 0.0 and abs(r["a"]) <= 0.5
    c, m = RL.clip_stats(ctx.rv * np.exp(-((1 - ctx.lv.pi) * ctx.lv.lnb + ctx.lv.pi * np.log(ctx.x["C"].to_numpy(float)) + ctx.lv.bg)), P)
    assert (r["c"], r["m"]) == (c, m)
    ref = minimize_scalar(lambda a: np_loss(a, ctx, P, "pi", c, m, r["lam"]), bounds=(-0.5, 0.5), method="bounded", options=dict(xatol=1e-12)).x
    assert abs(r["a"] - ref) < 1e-6
    assert np.allclose(r["C"], ctx.x["C"].to_numpy(float)[1000:1020] * np.exp(r["a"]), rtol=0, atol=0)


def test_validation_uses_true_pi_for_all_three_kinds_independent_recount():
    ctx = S.syn_ctx("F", ths=SH)
    lv, c8 = ctx.lv, ctx.x["C"].to_numpy(float)
    for kind in ML.KINDS:
        lam, w, G = LW.choose_l(ctx, P, kind)
        for bi, (T, Vb) in enumerate(W.blocks(P)):
            c, m = ML.stats_l(kind, LW.arrays(ctx), T)
            for l2 in W.LAMS:
                a = minimize_scalar(lambda z: np_loss(z, ctx, T, kind, c, m, l2), bounds=(-0.5, 0.5), method="bounded", options=dict(xatol=1e-12)).x
                f = lambda aa: np.exp((1 - lv.pi[Vb]) * lv.lnb[Vb] + lv.pi[Vb] * (np.log(c8[Vb]) + aa) + lv.bg[Vb])
                want = float(RL.ell(ctx.rv[Vb], np.log(f(a))).mean() - RL.ell(ctx.rv[Vb], np.log(f(0.0))).mean())
                assert abs(want - G[l2][bi]) < 1e-6, (kind, bi, l2, want, G[l2][bi])


def test_pit_future_rv_and_pairs_after_pool_do_not_change_month_and_plus_one_does():
    ctx = S.syn_ctx("F", ths=SH)
    D2 = copy.copy(ctx.D)
    D2.rv = ctx.D.rv.copy()
    D2.rv[1000:] = np.random.default_rng(1).uniform(1e-3, 1e3, len(D2.rv) - 1000)
    ctx2 = W.Ctx("CK", D2, ctx.lv, ctx.x, None)
    for kind in ML.KINDS:
        assert same(month(ctx, kind), month(ctx2, kind))
    plus1 = np.arange(1001)    # «закрыто + 1»: пара t⁺ > D в пуле
    assert any(not same(month(ctx, k), month(ctx2, k, plus1)) for k in ML.KINDS)


def test_real_pool_is_exactly_pairs_with_tplus_le_D_and_d_le_D_mutant_differs():
    x = C.load()
    xd, tp = pd.DatetimeIndex(x["d"]), pd.DatetimeIndex(x["t_plus"])
    mb = W.month_bounds(type("Cx", (), {"x": x, "arch": "CK"})())
    bad_mut = 0
    for a, _ in mb[::9]:
        p = RL.pool_of("CK", None, x, a)
        want = np.flatnonzero(np.asarray(tp <= xd[a]))
        assert set(p) == set(want) and (tp[p].diff()[1:] >= pd.Timedelta(0)).all()
        bad_mut += set(np.flatnonzero(np.asarray(xd <= xd[a]))) != set(want)
    assert bad_mut > 0
