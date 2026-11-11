"""S-Z-IMPL-2 (Ж-258), тестировщик: мутации процедуры R в `z_walk.choose` / `walk` (mutations.md № 144, 164, 146; spec разд. 8) — правило выбора, θ = 0 в сетке, блоки T_b ∩ V_b = ∅,
валидация обычной QLIKE, «закрыто + 1», z дня d + 1 в прогнозе, энкодер года Y и z пары от энкодера её года. Каждая мутация роняет детектор на боевом пути (`choose` / `walk`)."""
import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from models_v2.s16 import x1_ya_walk as YA
from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_walk as RW
from models_v2.s18 import r_walk_syn as RS
from models_v2.s18 import z_model as ZM
from models_v2.s18 import z_walk as ZW

L1 = RW.LAMS[0]
NP_, ORIG_BLOCKS = 800, RW.blocks


def stub_choose(monkeypatch, G):
    """choose с подставленными G (λ → [b1, b2, b3]): заглушки подгонки и потери; возвращает записи (max T, min V) на каждую пару."""
    rec = []
    st = SimpleNamespace(x0=np.ones(NP_), rv=np.ones(NP_), lnf0=np.zeros(NP_), view=lambda s: np.asarray(s))
    monkeypatch.setattr(ZW, "fit_heads", lambda VT, zts, var, use_z, lam, c, m: [(lam, int(VT.max()))] * len(zts))

    def lnc(ths, V, zts, var, use_z):
        lam, tmax = ths[0]
        b = (int(V[0]) - (NP_ - 3 * RW.VAL)) // RW.VAL
        rec.append((tmax, int(V.min())))
        return np.full(len(V), G[lam][b])
    monkeypatch.setattr(ZW, "head_lnc", lnc)
    monkeypatch.setattr(ZW.RL, "ln_f", lambda c, V: c)
    monkeypatch.setattr(ZW.RL, "ell", lambda rv, lnf: np.asarray(lnf, float))
    zy = [np.random.default_rng(s).normal(size=(NP_, 4)) for s in range(5)]
    return lambda: (*ZW.choose(st, zy, np.arange(NP_), "Z", True)[:2], rec)


def problems(monkeypatch):
    bad = []
    for i, (G, want) in enumerate(RS.BLOCK_CASES):
        run = stub_choose(monkeypatch, G)
        lam, w, rec = run()
        bad += ["случай %d: λ* %s ≠ %s" % (i, lam, want)] * (lam != want or w != {k: sum(g < 0 for g in G[k]) for k in G})
        bad += ["T ∩ V ≠ ∅ (случай %d)" % i] * any(tm >= vm for tm, vm in rec)
    return bad


def test_choose_follows_rule_two_of_three_zero_in_grid_blocks_disjoint(monkeypatch):
    assert problems(monkeypatch) == []


def test_mut_rule_one_of_three_is_caught(monkeypatch):
    monkeypatch.setattr(RW, "MIN_W", 1)
    assert problems(monkeypatch) != []


def test_mut_theta_zero_absent_from_grid_is_caught(monkeypatch):
    def mut(G, rule):
        w = {lam: sum(g < 0 for g in G[lam]) for lam in RW.LAMS}
        return max(RW.LAMS, key=lambda lam: (w[lam], lam)), w
    monkeypatch.setattr(RW, "decide", mut)
    assert problems(monkeypatch) != []


def test_mut_training_includes_validation_block_is_caught(monkeypatch):
    monkeypatch.setattr(RW, "blocks", lambda P, nb=RW.NB: [(np.asarray(P)[: len(P) - RW.VAL * (nb - b - 1)], V) for b, (T, V) in enumerate(ORIG_BLOCKS(P, nb))])
    assert problems(monkeypatch) != []


def val_gap():
    """G блока V₃ при λ = 0,1 против независимой обычной QLIKE (в V₃ есть выброс x = 40)."""
    base = RS.syn_ctx(1100, 3, ths=np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.3]), spike=950)
    zy = [np.random.default_rng(s).normal(size=(1100, 4)) for s in range(5)]
    P, T, V = np.arange(1000), np.arange(874), np.arange(874, 1000)
    G = ZW.choose(base, zy, P, "Z", True)[2][L1][2]
    c, m, zt = *RL.clip_stats(base.x0, T), ZW._zts(zy, T)
    heads = [ZM.fit_head(base.view(T), z[T], "Z", L1, c, m, "R") for z in zt]
    lnf = RL.ln_f(torch.as_tensor(ZM.predict_lnc(heads, base.view(V), [z[V] for z in zt], "Z")), base.view(V)).numpy()
    q = lambda lf: float(np.mean(base.rv[V] * np.exp(-lf) + lf - np.log(base.rv[V]) - 1.0))
    return abs(G - (q(lnf) - q(base.lnf0[V])))


def test_validation_is_plain_qlike_and_mutation_rho_kappa_is_caught(monkeypatch):
    assert val_gap() < 1e-12
    orig = RL.ell
    monkeypatch.setattr(RL, "ell", lambda rv, lnf: np.minimum(orig(rv, lnf), 0.3))
    assert val_gap() > 1e-6


# ---- walk: закрытый пул, z дня d + 1, энкодер года ----
def walk_ctx(per_year=False, seed=6):
    base = RS.syn_ctx(1500, seed, ths=np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.3]))
    rng = np.random.default_rng(seed)
    years = sorted({int(d.year) for d in pd.DatetimeIndex(base.x["d"])})
    zsrc = {y: {s: rng.normal(size=(1500, 4)) for s in ZW.SEEDS} for y in years}
    if not per_year:
        zsrc = {y: zsrc[years[0]] for y in years}
    a, b = RW.month_bounds(base)[40]     # после включения: пул = 40 месяцев · ~21 пар > 630
    return base, zsrc, a, b


def one(ctx, zsrc, a):
    return ZW.walk(ctx, zsrc, "Z", True, positions={a}, quiet=True)[0][2]


def damage(ctx, zsrc, a, b):
    D = copy.copy(ctx.D)
    D.rv = D.rv.copy()
    D.rv[a:] *= 50.0
    x = ctx.x.copy()
    x["rv"] = x["rv"].to_numpy(float).copy()
    x.loc[x.index[a:], "rv"] *= 50.0
    zb = {y: {s: np.where((np.arange(len(z)) >= b)[:, None], 9.0 * z + 40.0, z) for s, z in d.items()} for y, d in zsrc.items()}
    return RW.Ctx("CK", D, ctx.lv, x, ctx.gap, ctx.dp), zb


def pit_ok(ctx, zsrc, a, b):
    r0, (c1, z1) = one(ctx, zsrc, a), damage(ctx, zsrc, a, b)
    r1 = one(c1, z1, a)
    same = all(np.array_equal(np.asarray(r0[k]), np.asarray(r1[k]), equal_nan=True) for k in ("lam", "w", "G", "c", "m", "theta", "thz", "C"))
    return same and r0["n_train"] == a


def test_walk_pit_pool_and_forecast_ignore_future_and_pool_is_closed_pairs():
    ctx, zsrc, a, b = walk_ctx()
    assert pit_ok(ctx, zsrc, a, b) and one(ctx, zsrc, a)["lam"] >= 0.0


def test_mut_closed_plus_one_is_caught(monkeypatch):
    ctx, zsrc, a, b = walk_ctx()
    assert pit_ok(ctx, zsrc, a, b)
    monkeypatch.setattr(YA, "closed_date", lambda tp, d: d)
    assert not pit_ok(ctx, zsrc, a, b)


def test_mut_forecast_with_z_of_day_d_plus_1_is_caught(monkeypatch):
    ctx, zsrc, a, b = walk_ctx()
    assert pit_ok(ctx, zsrc, a, b)
    box, o_zts, o_hl, days = {}, ZW._zts, ZW.head_lnc, np.arange(a, b)
    monkeypatch.setattr(ZW, "_zts", lambda zy, T, zero=False: box.update(t=o_zts(zy, T, zero)) or box["t"])
    monkeypatch.setattr(ZW, "head_lnc", lambda ths, V, zts, var, use_z: o_hl(ths, V, [t[days + 1] for t in box["t"]] if len(zts[0]) == len(days) else zts, var, use_z))
    assert not pit_ok(ctx, zsrc, a, b)


def spy_inputs(monkeypatch, ctx, zsrc, a):
    seen, o = [], ZW._zts
    monkeypatch.setattr(ZW, "_zts", lambda zy, T, zero=False: (seen.append(zy), o(zy, T, zero))[1])
    one(ctx, zsrc, a)
    Y = pd.Timestamp(ctx.x["d"].iloc[a]).year
    return seen and all(len(zy) == 5 and all(np.array_equal(zy[i], zsrc[Y][s]) for i, s in enumerate(ZW.SEEDS)) for zy in seen)


def test_encoder_year_is_year_of_D_for_pool_and_forecast_and_mutations_are_caught(monkeypatch):
    ctx, zsrc, a, b = walk_ctx(per_year=True)
    Y, orig = pd.Timestamp(ctx.x["d"].iloc[a]).year, ZW.fit_month
    assert spy_inputs(monkeypatch, ctx, zsrc, a)
    nxt = lambda zy: [zsrc[Y + 1][s] for s in ZW.SEEDS]
    own = lambda zy: [np.vstack([zsrc[Y - 1][s][:a], zsrc[Y][s][a:]]) for s in ZW.SEEDS]      # z пар прошлых лет — от энкодера их года (№ 146)
    for mut in (nxt, own):
        with monkeypatch.context() as mp:
            mp.setattr(ZW, "fit_month", lambda ctx_, zy, *r, _m=mut, **k: orig(ctx_, _m(zy), *r, **k))
            assert not spy_inputs(mp, ctx, zsrc, a)
