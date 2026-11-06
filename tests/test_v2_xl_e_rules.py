"""Волна E Ж-206 (GATES_E G2): правила весов используют только закрытые прогнозы; мутации «закрыто+1», «fixed по всей выборке», «Холм без N», границы ACCEPT."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s5.d import ens, stats_d
from models_v2.s6.e import crit_e, pe, rules

N = 1000
RULES = ("fixed_loyo", "inv_qlike_252", "inv_rmse_252", "shrink_252", "dyn_63", "dyn_126", "dyn_252", "stack_nnls", "roll_252_con", "reg_g4", "roll_252",
         "roll_126", "roll_504")


def synth(seed=3, k=2):
    rng = np.random.default_rng(seed)
    d = np.datetime64("2017-03-01") + np.arange(N).astype("timedelta64[D]")
    rv = np.exp(rng.normal(0, 0.4, N))
    S = np.column_stack([rv * np.exp(rng.normal(0, s, N)) for s in (0.3, 0.5, 0.7)[:k]])
    ctx = {"rv": rv, "d": d.astype("datetime64[ns]"), "td": (d + 1).astype("datetime64[ns]"), "st_g4": rng.random(N) > 0.6}
    return ctx, S


def perturb(ctx, S, k0):
    c = {**ctx, "rv": ctx["rv"].copy()}
    S2 = S.copy()
    c["rv"][k0:] *= 7.0
    S2[k0:] *= 3.0
    return c, S2


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("K", (2, 3))
def test_weight_ignores_unclosed_future(rule, K):
    ctx, S = synth(k=K)
    k0 = 430
    c2, S2 = perturb(ctx, S, k0)
    a, b = rules.weights(rule, S, ctx)[:k0 + 1], rules.weights(rule, S2, c2)[:k0 + 1]
    assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("rule", [r for r in RULES if r != "fixed_loyo"])
def test_mutation_closed_plus_one_is_caught(rule):
    ctx, S = synth()
    k0 = 441                            # кратно 21: переоценка stack_nnls
    c2, S2 = perturb(ctx, S, k0)
    S2[k0, 1] *= 1e30                   # первый НЕзакрытый прогноз испорчен: честное правило его не видит, мутация extra=1 — видит
    assert rules.weights(rule, S, ctx)[k0].tobytes() == rules.weights(rule, S2, c2)[k0].tobytes()
    assert rules.weights(rule, S, ctx, extra=1)[k0].tobytes() != rules.weights(rule, S2, c2, extra=1)[k0].tobytes()


def test_mutation_fixed_loyo_year_plus_one_is_caught():
    """Мутация «target < 01.01.Y+1»: вес года Y видит прогнозы года Y — порча внутри года Y меняет вес этого года."""
    ctx, S = synth()
    y = pd.Series(ctx["d"]).dt.year.to_numpy()
    kk = int(np.flatnonzero(y == 2019)[0]) + 40      # внутри 2019: вес 2019 фиксирован по целям < 01.01.2019; порча с kk его не меняет
    c2, S2 = perturb(ctx, S, kk)
    S2[kk:, 0] *= 20.0
    S2[kk:, 1] *= 0.05
    for ey, same in ((0, True), (1, False)):          # мутация extra_year=1 (до 01.01.2020) видит прогнозы самого 2019 года
        a, b = rules.weights("fixed_loyo", S, ctx, extra_year=ey)[kk], rules.weights("fixed_loyo", S2, c2, extra_year=ey)[kk]
        assert (a.tobytes() == b.tobytes()) is same, (ey, a, b)


def test_fixed_loyo_equal_in_first_year_and_constant_within_year():
    ctx, S = synth()
    w = rules.weights("fixed_loyo", S, ctx)
    y = pd.Series(ctx["d"]).dt.year.to_numpy()
    assert (w[y == 2017] == 0.5).all()
    for Y in (2018, 2019):
        assert (w[y == Y] == w[y == Y][0]).all()


def test_shrink_s_chosen_only_from_closed_and_mutation_caught():
    ctx, S = synth()
    ctx2, S2 = perturb(ctx, S, 460)
    k0 = 460
    assert rules.weights("shrink_252", S, ctx)[k0].tobytes() == rules.weights("shrink_252", S2, ctx2)[k0].tobytes()
    assert rules.w_shrink(ctx["rv"], S, ens.closed(ctx), extra_sel=0).shape == S.shape
    S3 = S.copy()
    S3[k0:k0 + 3, 0] *= 1e6
    m, rows = ens.closed(ctx), slice(k0 - 6, k0 + 1)
    assert rules.w_shrink(ctx["rv"], S3, m)[rows].tobytes() == rules.w_shrink(ctx["rv"], S, m)[rows].tobytes()
    assert rules.w_shrink(ctx["rv"], S3, m, extra_sel=6)[rows].tobytes() != rules.w_shrink(ctx["rv"], S, m, extra_sel=6)[rows].tobytes()


def test_shrink_extremes_are_roll_and_equal():
    ctx, S = synth()
    m = ens.closed(ctx)
    r, e = ens.w_window(ctx["rv"], S, m, 252), rules.w_equal(S)
    w0 = rules.w_shrink(ctx["rv"], S, m, grid=(0.0,))
    w1 = rules.w_shrink(ctx["rv"], S, m, grid=(1.0,))
    k = int(np.flatnonzero(m >= 252 + 126)[0])
    assert np.allclose(w0[k], r[k]) and np.allclose(w1, e)


def test_weights_nonnegative_and_simplex_sums():
    ctx, S = synth(k=3)
    for r in RULES:
        w = rules.weights(r, S, ctx)
        assert (w >= 0).all() and np.isfinite(w).all()
        if r != "stack_nnls":
            assert np.allclose(w.sum(axis=1), 1.0)


def test_median_of_two_equals_mean_exactly():
    _, S = synth(k=2)
    assert rules.sigma2("median", S, {}).tobytes() == rules.sigma2("mean_lvl", S, {}).tobytes()


def test_fixed_loyo_not_whole_sample():
    ctx, S = synth()
    c2, S2 = perturb(ctx, S, 430)
    S2[430:, 0] *= 50.0
    y = pd.Series(ctx["d"]).dt.year.to_numpy()
    sel = (y == 2018) & (np.arange(N) <= 430)
    assert rules.weights("fixed_loyo", S, ctx)[sel].tobytes() == rules.weights("fixed_loyo", S2, c2)[sel].tobytes()
    G = ens.simplex(2)
    whole = ens.pick(ens.cum_qlike(ctx["rv"], S, G)[-1], G)          # мутация: оптимум по всей выборке — зависит от порчи после границы
    assert not np.array_equal(whole, ens.pick(ens.cum_qlike(c2["rv"], S2, G)[-1], G))


def test_known_answer_exact_member_wins():
    ctx, S = synth()
    rng = np.random.default_rng(5)
    S = np.column_stack([ctx["rv"] * np.exp(rng.normal(0, 0.05, N)), ctx["rv"] * 3.0 * np.exp(rng.normal(0, 0.05, N))])
    for r in ("dyn_126", "roll_252", "inv_qlike_252"):
        w = rules.weights(r, S, ctx)[-1]
        assert w[0] > 0.85 and w[0] > w[1], (r, w)
    ws = rules.weights("stack_nnls", S, ctx)[-1]                  # S0 ≈ RV, S1 ≈ 3·RV коллинеарны: единственности весов нет, смесь обязана быть ≈ RV
    q = lambda s2: rules.qlike_mean(ctx["rv"], s2)   # noqa: E731
    assert q(S @ ws) < 0.5 * q(S.mean(axis=1)) and (ws >= 0).all()
    assert np.allclose(rules.weights("roll_252_con", S, ctx)[-1], [0.9, 0.1])     # ограничение 0,1


def test_stack_gradient_matches_finite_differences():
    ctx, S = synth(k=3)
    from scipy.optimize import approx_fprime
    rv = ctx["rv"][:300]

    def f(w):
        s2 = S[:300] @ w
        r = rv / s2
        return float((r - np.log(r) - 1).mean())
    w0 = np.array([0.5, 0.3, 0.4])
    g = ((1.0 / (S[:300] @ w0) - rv / (S[:300] @ w0) ** 2)[:, None] * S[:300]).mean(axis=0)
    assert np.allclose(g, approx_fprime(w0, f, 1e-7), atol=1e-5)


def test_holm_uses_n_and_rejects_small_n():
    p = np.array([0.001, 0.02, 0.03])
    assert (stats_d.holm_n(p, 100) > stats_d.holm_n(p, 3)).all()          # мутация «Холм без N» (N = число тестов) даёт меньшие p
    with pytest.raises(ValueError):
        stats_d.holm_n(p, 2)


def test_n_trials_counts_registries(tmp_path, monkeypatch):
    for a, n in (("xgboost", 3), ("lightgbm", 2), ("xgb_lgbm_ensemble", 4)):
        (tmp_path / a).mkdir()
        pd.DataFrame({"trial_id": ["%s%d" % (a[:2], i) for i in range(n)]}).to_csv(tmp_path / a / "OPTUNA_TRIALS.csv", index=False)
    monkeypatch.setattr(pe, "REP", tmp_path)
    monkeypatch.setattr(pe, "TRIALS", tmp_path / "xgb_lgbm_ensemble" / "OPTUNA_TRIALS.csv")
    assert pe.n_trials() == 9


def test_register_limits_and_duplicate(tmp_path, monkeypatch):
    monkeypatch.setattr(pe, "ENS", tmp_path)
    monkeypatch.setattr(pe, "TRIALS", tmp_path / "OPTUNA_TRIALS.csv")
    monkeypatch.setattr(pe, "CHECK", tmp_path / "CHECKPOINT.json")
    t = [pe.trial({"i": i}, "EA", "ensemble", "m") for i in range(22)]
    assert pe.register(t) == 22 and pe.register(t) == 22          # дубль — не новый trial
    with pytest.raises(ValueError):
        pe.register([pe.trial({"i": 99}, "EA", "ensemble", "m")])  # 23-й trial типа A
    with pytest.raises(ValueError):
        pe.register([pe.trial({"i": 98}, "E1", "ensemble", "m")])  # чужая волна
    assert pe.register([pe.trial({"i": 100 + j}, "EB", "residual", "m") for j in range(9)]) == 31
    with pytest.raises(ValueError):
        pe.register([pe.trial({"i": 200}, "EB", "residual", "m")])  # 10-й trial типа B


def _acc(**kw):
    base = {"qlike_2017_2021": 0.1600}
    new = {"qlike_2017_2021": kw.get("q17", 0.1600)}
    td = pd.Series(pd.date_range("2017-01-02", periods=2610, freq="B"))
    big = td.dt.year.isin(kw.get("big", ()))
    d = np.where(big, -0.05, kw.get("rest", -0.01))                              # выигрыш держат «большие» годы, в остальных rest
    return crit_e.accept(new, base, d, np.zeros(len(d)), td, kw.get("sn", [0.1]), kw.get("sb", [0.2]), 0.1, 0.2)


def test_accept_boundaries():
    assert _acc()["accept"] is True and _acc()["years_won"] == 10
    r9 = _acc(big=(2022,), rest=0.001)                  # один год держит выигрыш: без него знак меняется — 9 из 10 хватает
    assert r9["years_won"] == 9 and r9["accept"] is True
    r8 = _acc(big=(2020, 2022), rest=0.01)               # два года держат выигрыш — 8 из 10
    assert r8["years_won"] == 8 and r8["accept"] is False
    assert _acc(q17=0.1600 * 1.02)["accept"] is False                           # 2017–2021 хуже более чем на 1 %
    assert _acc(q17=0.1600 * 1.0099)["accept"] is True
    assert _acc(sn=None)["accept"] is False                                     # seed не проверены
    assert _acc(sn=[0.3], sb=[0.2])["accept"] is False


def test_c_rules_ignore_unclosed_future_and_mutation_caught():
    ctx, S = synth()
    k0 = 441
    c2, S2 = perturb(ctx, S, k0)
    S2[k0, 0] *= 1e30
    m, m1 = ens.closed(ctx), ens.closed(ctx, 1)
    g = np.random.default_rng(9).integers(0, 3, N)
    funcs = {"group": lambda r, s, mm: rules.w_group(r, s, mm, g), "switch": lambda r, s, mm: rules.w_switch(r, s, mm),
             "shrink_c": lambda r, s, mm: rules.c_shrink(r, s[:, 1], s[:, 0], mm)}
    for nm, f in funcs.items():
        assert f(ctx["rv"], S, m)[:k0 + 1].tobytes() == f(c2["rv"], S2, m)[:k0 + 1].tobytes(), nm            # будущее не видно
        assert f(ctx["rv"], S, m1)[k0].tobytes() != f(c2["rv"], S2, m1)[k0].tobytes(), nm                    # мутация «закрыто+1» видна


def test_switch_known_answer_and_c_shrink_bounds():
    ctx, S = synth()
    S = np.column_stack([ctx["rv"] * 3.0, ctx["rv"] * 1.05])         # HAR плох, ML хорош
    m = ens.closed(ctx)
    w = rules.w_switch(ctx["rv"], S, m)
    assert (w[200:] == [0.0, 1.0]).all() and (w[:50] == [1.0, 0.0]).all()
    c = rules.c_shrink(ctx["rv"], S[:, 0], S[:, 1], m)
    assert (c[300:] == 1.0).all() and (c[:100] == 0.5).all() and ((c >= 0) & (c <= 1)).all()
    c0 = rules.c_shrink(ctx["rv"], S[:, 1], S[:, 0], m)              # коррекция всегда хуже HAR -> c = 0
    assert (c0[300:] == 0.0).all()
