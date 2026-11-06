"""Ж-214, лист 3, V7/V11: декомпозиции S/J/T, J3, согласование. Синтетика с известным ответом, главный тест обрезки на каждой конфигурации, мутации темы:
части дня не в сумме, нулевая компонента, отрицательная компонента в согласовании, закрыто+1 в весах согласования, будущие пары, J3 с будущими парами. Без данных и сети."""
import numpy as np
import pandas as pd
import pytest

from models_v2 import har as H
from models_v2.s9 import hv2_dec as DC, hv2_decstage as DS, hv2_jlogit as JL, hv2_rec as R, hv2_recstage as RS, hv2_world as W


def synth(n=700, seed=5, tail=420):
    rng = np.random.default_rng(seed)
    ln = np.zeros(n)
    for t in range(1, n):
        ln[t] = 0.85 * ln[t - 1] + rng.normal(0, 0.4)
    rv = np.exp(ln + 1.0)
    u, sh = rng.uniform(0.3, 0.7, n), rng.dirichlet([4, 8, 4], n)
    j = rv * rng.uniform(0, 0.35, n) * (rng.random(n) < 0.6)
    m10 = pd.DataFrame({"rv": rv, "j": j, "c": rv - j, "rs_pos": rv * u, "rs_neg": rv * (1 - u), "rv_open": rv * sh[:, 0], "rv_mid": rv * sh[:, 1], "rv_close": rv * sh[:, 2]},
                       index=pd.bdate_range("2018-01-01", periods=n))
    seg = np.zeros(n, int)
    X = np.column_stack([np.log(rv), np.log(DC.wmean(rv, seg, 5)), np.log(DC.wmean(rv, seg, 22)), np.log1p(j)])
    p = H.Panel(m10.index.to_numpy("datetime64[ns]"), seg, rv, X)
    rq = rv ** 2
    w = W.World(p, rq, np.ones(n), H.eligible(p)[-tail:])
    return DC.build(w, m10), m10


def test_identities_of_actuals_and_forecast_sums():
    w, m10 = synth()
    assert np.allclose(m10.rv_open + m10.rv_mid + m10.rv_close, m10.rv, rtol=1e-12) and np.allclose(m10.c + m10.j, m10.rv, rtol=1e-12)
    for cfg in DC.CONFIGS:
        s2, comps, fb = DC.run(w, cfg)
        idn = DC.identities(s2, comps, fb)
        assert idn["min_sigma2"] > 0 and np.isfinite(s2).all() and idn["max_rel_sum_err"] < 1e-12, (cfg, idn)
        assert idn["min_component"] > 0 or (cfg == "J1" and idn["min_component"] >= 0)
        assert idn["n_fallback"] < 60


def test_mutation_parts_not_in_sum_is_caught():
    w, _ = synth()
    s2, comps, fb = DC.run(w, "T1")
    cut = {k: comps[k] for k in ("rv_open", "rv_mid")}          # МУТАНТ: часть дня (CLOSE) не в сумме
    assert DC.identities(s2, cut, fb)["max_rel_sum_err"] > 0.05


def test_forecast_known_answer_noiseless():
    rng = np.random.default_rng(1)
    w, _ = synth(n=300, tail=100)
    F = rng.normal(size=(300, 2))
    y = 0.1 + 0.5 * F[:, 0] + 0.2 * F[:, 1]
    tgt = np.r_[np.nan, y[:-1]]                      # y_{t+1} = g(F_t): цель сдвинута на день
    out = DC.forecast(w, np.vstack([F, F[:0]]), tgt)
    assert np.allclose(out, np.exp(y[w.el]), rtol=1e-8)


def test_zero_component_gives_fallback_to_a0_not_inf():
    w, m10 = synth()
    d = int(w.el[-5])
    c = {k: v.copy() for k, v in w.comp.items()}
    c["rv_open"][d] = 0.0
    w0 = DC.dataclasses.replace(w, comp=c)
    a0 = DC.a0_forecast(w0)
    s2, comps, fb = DC.run(w0, "T1", a0)
    k = int(np.flatnonzero(w0.el == d)[0])
    assert fb[k] and s2[k] == a0[k] and np.isfinite(s2).all() and (s2 > 0).all()
    assert fb.sum() >= 1


def test_j_forecast_nonnegative_without_jumps():
    w, _ = synth()
    c = {k: v.copy() for k, v in w.comp.items()}
    c["j"][:] = 0.0
    X = w.p.X.copy()
    X[:, 3] = 0.0
    w0 = DC.dataclasses.replace(w, comp=c, p=H.Panel(w.p.dates, w.p.seg, w.p.rv, X))
    s2, comps, fb = DC.run(w0, "J1")
    assert (comps["j"][~fb] >= 0).all() and (s2 > 0).all()


@pytest.mark.parametrize("cfg", ["S1", "S3", "J1", "T1", "T3"])
def test_main_truncation_each_config(cfg):
    w, _ = synth(n=400, tail=150)
    i0 = int(w.el[90])
    assert DC.trunc_diff(lambda x: DC.sigma(x, cfg), w, i0) == 0.0


def test_main_truncation_catches_future_pairs(monkeypatch):
    w, _ = synth(n=400, tail=150)
    orig = W.pairs_idx
    monkeypatch.setattr(W, "pairs_idx", lambda seg, F, i: orig(seg, F, min(i + 3, len(seg) - 1)))    # МУТАНТ: будущие пары
    assert DC.trunc_diff(lambda x: DC.sigma(x, "S1"), w, int(w.el[90])) > 0


def test_corrupt_leaves_past_and_breaks_future():
    w, _ = synth(n=300, tail=100)
    i0 = int(w.el[40])
    c = DC.corrupt(w, i0)
    assert all((c.comp[k][:i0 + 1] == w.comp[k][:i0 + 1]).all() and (c.comp[k][i0 + 1:] != w.comp[k][i0 + 1:]).any() for k in w.comp)
    assert (c.comp["j"][i0 + 1:] == 0).any()


def test_tables_helpers_run_on_synthetic():
    w, _ = synth()
    s2, comps, fb = DC.run(w, "T3")
    ser = {"T3": s2, "T3__fb": fb.astype(float), **{"T3__c_%s" % k: v for k, v in comps.items()}}
    t = DS.comp_table("T3", ser, w)
    assert len(t) == 3 and (t["min_forecast"] > 0).all() and np.isfinite(t["rmse"]).all()
    df = pd.DataFrame({"target_date": w.td, "rv": w.target})
    y = DS.by_year(df, {"A0": s2 * 1.1, "XJ": s2 * 0.9, "T3": s2})
    assert set(y["model"]) == {"XJ", "T3"} and np.isfinite(y["d_vs_A0"]).all()


# ---------- J3 ----------
def test_fit_logit_known_answer_and_ridge_shrinks():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(20000, 2))
    y = (rng.random(20000) < 1 / (1 + np.exp(-(0.3 + 1.0 * Z[:, 0] - 0.5 * Z[:, 1])))).astype(float)
    b = JL.fit_logit(Z, y, lam=1e-6)
    assert np.allclose(b, [0.3, 1.0, -0.5], atol=0.06)
    assert np.abs(JL.fit_logit(Z, y, lam=5000.0)[1:]).max() < np.abs(b[1:]).max()


def test_auc_known_answers():
    y = np.array([0, 0, 1, 1])
    assert JL.auc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0 and JL.auc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 0.0
    assert JL.auc(np.ones(4), y) == 0.5 and np.isnan(JL.auc(np.arange(3.0), np.ones(3)))


def test_j3_forecast_in_unit_interval_and_metrics_skill_zero_for_base():
    w, _ = synth()
    ph, base, npair = JL.run(w)
    ok = np.isfinite(ph)
    assert ok.sum() > 300 and ((ph[ok] > 0) & (ph[ok] < 1)).all() and (npair[ok] >= 60).all()
    m = JL.metrics(w, base, base)
    assert abs(m.loc[m.period == "all", "brier_skill"].iloc[0]) < 1e-12
    assert set(m["period"]) >= {"all", "ex_240222"}


def test_j3_main_truncation_and_future_pairs_mutant():
    w, _ = synth(n=400, tail=150)
    i0 = int(w.el[90])
    assert DC.trunc_diff(lambda x: JL.run(x)[0], w, i0) == 0.0
    assert DC.trunc_diff(lambda x: JL.run(x, extra=1)[0], w, i0) > 0       # МУТАНТ: пара с целью d+1 в обучении


def test_j3_diagnostic_is_separate_from_forecast():
    _, m10 = synth()
    m10["grid_d"] = True
    t = JL.diagnostic(m10)
    assert {"share_days_J_pos", "mean_J_share_of_rv"} <= set(t.columns) and (t["share_days_J_pos"].between(0, 1)).all()


# ---------- согласование ----------
def _td(n):
    d = pd.bdate_range("2018-01-01", periods=n).to_numpy("datetime64[ns]")
    return d, np.r_[d[1:], d[-1] + np.timedelta64(1, "D")]


def test_blend_picks_better_forecast_after_252_closed_and_base_before():
    n = 400
    rv = np.exp(np.random.default_rng(3).normal(1, 0.3, n))
    d, td = _td(n)
    f, wd = R.blend(2 * rv, rv, rv, d, td)                       # f_Σ точный, f_д вдвое завышен
    assert (wd[:252] == 0).all() and (wd[300:] == 1.0).all() and np.allclose(f[300:], rv[300:])
    f2, wd2 = R.blend(rv, 2 * rv, rv, d, td)
    assert (wd2[300:] == 0).all()
    _, wd3 = R.blend(rv * 1.5, rv * 1.5, rv, d, td)              # ничья: w ближе к 0,5
    assert (wd3[300:] == 0.5).all()


def test_blend_closed_plus_one_mutant_uses_future():
    n = 400
    rv = np.exp(np.random.default_rng(4).normal(1, 0.3, n))
    d, td = _td(n)
    fd = fs = rv * 1.5                                           # до строки 300 прогнозы совпадают: ничья, w = 0,5
    fs = fs.copy()
    fs[300] = rv[300]                                            # строка 300 закрывается только на дне 301; f_Σ на ней точный, f_д нет
    _, a = R.blend(fd, fs, rv, d, td)
    _, b = R.blend(fd, fs, rv, d, td, extra=1)
    assert a[300] == 0.5 and b[300] == 1.0                       # закрыто+1 берёт незакрытую цель и меняет w


def test_reconcile_sum_positivity_zero_extreme_and_negative_mutant():
    comps = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [1e-12, 1e12, 0.0], [5.0, 0.0, 5.0]])
    f = np.array([12.0, 3.0, 7.0, 1e-9])
    rec, bad = R.reconcile(f, comps)
    assert np.allclose(rec.sum(axis=1), f, rtol=1e-12) and (rec >= 0).all()
    assert bad.tolist() == [False, True, False, False] and np.allclose(rec[1], 1.0)
    with pytest.raises(ValueError):
        R.reconcile(f[:1], np.array([[-1.0, 3.0, 4.0]]))          # МУТАНТ: отрицательная компонента на входе


def test_r2_run_main_truncation_and_extra_reaches_blend(monkeypatch):
    w, _ = synth(n=700, tail=420)
    pos = [int(w.el[k]) for k in range(300, 400, 25)]
    assert max(DC.trunc_diff(lambda x: R.sigma(x, "T1"), w, i0) for i0 in pos) == 0.0
    seen = []
    orig = R.blend
    monkeypatch.setattr(R, "blend", lambda *a, **k: (seen.append(a[5] if len(a) > 5 else k.get("extra", 0)), orig(*a, **k))[1])
    R.run_r2(DC.W.sub(w, int(w.el[60])), "T1", extra=1)
    assert seen == [1]                                            # «закрыто+1» доходит до выбора w (механизм ловится: test_blend_closed_plus_one_mutant_uses_future)


def test_r2_outputs_consistent_with_components():
    w, _ = synth(n=700, tail=420)
    f, wd, s2, a0, rec, flag = R.run_r2(w, "T3")
    ok = ~flag
    assert np.allclose(rec.sum(axis=1), f, rtol=1e-12) and (rec >= 0).all() and (f > 0).all() and ok.any()
    assert set(np.unique(wd)) <= set(R.GRID)
    t = RS.weights_table("R2T", wd, w)
    assert t["mean_w"].between(0, 1).all()
