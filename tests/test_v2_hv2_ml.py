"""Ж-214, лист 4, V9: структурная QLIKE-коррекция (N1, N2) XGB и LGBM раздельно. Синтетика с известным ответом, главный тест обрезки на каждой конфигурации, мутации темы:
градиент/гессиан QLIKE, λ = 0 ≢ база, штраф μ не передан, закрыто+1 в λ_wf, будущие пары, компонента вне суммы N2, признак из d+1. Без сети; боевой путь — те же функции, что у стадии `ml`."""
import dataclasses

import numpy as np
import pytest

from models_v2.s9 import hv2_dec as DC, hv2_eval as E, hv2_ledger as L, hv2_mlout as O, hv2_mlstage as MLS, hv2_msfeat as MS, hv2_nfit as NF, hv2_paths as P, hv2_synth_ml as SY

MP, MC = 100, 15


synth = SY.dworld


def run1(dw, lib, tail=None, **kw):
    if tail:
        dw = dataclasses.replace(dw, el=dw.el[-tail:])
    B = MS.bases(dw, need=("A0",), min_pairs=MP)
    return B, NF.run_n1(dw, B["A0"], MS.GROUPS, lib, 0.0, 1, min_pairs=MP, min_closed=MC, **kw)


def test_derivatives_numeric_positive_and_mutations_caught():
    rng = np.random.default_rng(1)
    y, m, off = np.exp(rng.normal(0, 1, 200)), rng.normal(0, 1, 200), rng.normal(0, 0.3, 200)
    for mu in (0.0, 0.05):
        r = NF.deriv_check(y, m, off, mu)
        assert r["max_err_grad"] < 1e-7 and r["max_err_hess"] < 1e-6 and r["min_hess"] > 0 and r["min_f"] > 0
    bad_sign = lambda y_, m_, o_, mu_: (y_ * np.exp(-m_) - 1.0, y_ * np.exp(-m_))              # МУТАНТ: знак градиента
    bad_hess = lambda y_, m_, o_, mu_: (1.0 - y_ * np.exp(-m_), np.ones_like(m_))              # МУТАНТ: гессиан = 1
    assert NF.deriv_check(y, m, off, 0.0, bad_sign)["max_err_grad"] > 1e-2
    assert NF.deriv_check(y, m, off, 0.0, bad_hess)["max_err_hess"] > 1e-2


@pytest.mark.parametrize("lib", ["XGB", "LGBM"])
def test_lambda_zero_is_base_bitwise_lambda_one_is_full_on_run_path(lib, monkeypatch):
    dw = synth(signal=1.0)
    B, o = run1(dw, lib)
    base = B["A0"][dw.el]
    S = NF.n1_candidates(base, o["g"])
    assert np.array_equal(S[:, 0], base) and np.array_equal(o["full"], base * np.exp(o["g"])) and np.array_equal(S[:, -1], o["full"])
    z, one = o["lam"] == 0.0, o["lam"] == 1.0
    assert np.array_equal(o["s2"][z], base[z]) and np.array_equal(o["s2"][one], o["full"][one])
    monkeypatch.setattr(NF, "LAMBDAS", (0.01, 0.25, 0.5, 0.75, 1.0))              # МУТАНТ: «λ = 0» не нулевой
    assert not np.array_equal(NF.n1_candidates(base, o["g"])[:, 0], base)


@pytest.mark.parametrize("lib", ["XGB", "LGBM"])
def test_known_answer_signal_group_helps_and_sigma_positive(lib):
    dw = synth(n=700, tail=200, signal=1.0)
    B, o = run1(dw, lib)
    base, rv = B["A0"][dw.el], dw.p.rv[dw.el + 1]
    from models_v2 import metrics
    q_new, q_base = metrics.qlike(rv, o["s2"]).mean(), metrics.qlike(rv, base).mean()
    assert (o["s2"] > 0).all() and (o["full"] > 0).all() and q_new < q_base and (o["lam"] > 0).mean() > 0.2
    assert not o["fb"].any()


@pytest.mark.parametrize("lib", ["XGB", "LGBM"])
def test_main_truncation_n1_and_future_pairs_mutation(lib):
    dw = synth()
    i0 = int(dw.el[-20])
    f = lambda pe: (lambda x: run1(x, lib, tail=30, pair_extra=pe)[1]["s2"])
    assert DC.trunc_diff(f(0), dw, i0) == 0.0
    assert DC.trunc_diff(f(1), dw, i0) > 0.0                                       # МУТАНТ: пара с t+1 = d+1 в обучении


def test_lam_pick_closed_plus_one_mutation():
    n, k = 40, 30
    d = (np.datetime64("2018-01-01") + np.arange(n)).astype("datetime64[ns]")
    td = d + np.timedelta64(1, "D")
    S = np.column_stack([np.ones(n), np.full(n, 2.0), np.full(n, 4.0), np.full(n, 4.0), np.full(n, 4.0)])
    S[k, 0] = 1e-3                                                                 # исход строки k: λ = 0 катастрофически плох
    rv = np.ones(n)
    assert NF.lam_pick(rv, S, d, td, 5, 0)[1][k] == 0.0                            # исход k ещё не закрыт на d_k
    assert NF.lam_pick(rv, S, d, td, 5, 1)[1][k] == NF.LAMBDAS[1]                 # МУТАНТ «закрыто + 1» видит его — и меняет выбор
    assert (NF.lam_pick(rv, S, d, td, 300, 0)[1] == 0.0).all()                     # меньше 252 закрытых — база


def test_penalty_mu_shrinks_correction_and_mutation_mu_not_passed():
    dw = synth(signal=1.0)
    B = MS.bases(dw, need=("A0",), min_pairs=MP)
    lb = np.log(B["A0"])
    X, _ = MS.feature_matrix(dw, lb)
    y = NF.target(dw.p.rv)
    S0 = dict(X=X, y=y, lb=lb, seg=dw.p.seg, gate=X[:, :4], lib="XGB", min_pairs=MP)
    amp = {}
    for mu in (0.0, 5.0):
        a = []
        for i in dw.el[-12:]:
            g, fb = NF.fit_one({**S0, "mu": mu}, int(i))
            idx = NF.pair_rows(dw.p.seg, X[:, :4], lb, y, int(i))
            a.append(abs(g - np.log(np.mean(y[idx] / np.exp(lb[idx])))))
        amp[mu] = np.mean(a)
    assert amp[5.0] < 0.6 * amp[0.0]                                               # если μ не доходит до objective — равенство, тест падает


def test_n2_components_lambda_zero_is_t3_sum_fallback_and_truncation():
    dw = synth()
    B = MS.bases(dw, need=("A0", "T3"), min_pairs=MP)
    t3, el = B["T3"], dw.el
    o = NF.run_n2(dw, t3, MS.GROUPS, "XGB", 0.0, 1, min_pairs=MP, min_closed=MC)
    ks = list(t3["comps"])
    comps = [t3["comps"][k][el] for k in ks]
    S = NF.n2_candidates(comps, [o["g_" + k] for k in ks], t3["fb"][el] > 0, t3["s2"][el])
    assert np.array_equal(S[:, 0], t3["s2"][el]) and (o["s2"] > 0).all()
    ok = t3["fb"][el] == 0
    assert np.allclose(o["full"][ok], np.sum([c * np.exp(o["g_" + k]) for c, k in zip(comps, ks)], axis=0)[ok], rtol=1e-14)
    cut = [c for c in comps[:2]]                                                   # МУТАНТ: компонента CLOSE вне суммы
    assert not np.array_equal(NF.n2_candidates(cut, [o["g_" + k] for k in ks[:2]], t3["fb"][el] > 0, t3["s2"][el])[:, 0][ok], t3["s2"][el][ok])
    i0 = int(dw.el[-20])
    def f(x):
        x = dataclasses.replace(x, el=x.el[-30:])
        return NF.run_n2(x, MS.bases(x, need=("A0", "T3"), min_pairs=MP)["T3"], MS.GROUPS, "XGB", 0.0, 1, min_pairs=MP, min_closed=MC)["s2"]
    assert DC.trunc_diff(f, dw, i0) == 0.0


def test_n2_zero_component_day_uses_a0_not_inf():
    dw = synth()
    d = int(dw.el[-5])
    c = {k: v.copy() for k, v in dw.comp.items()}
    c["rv_open"][d] = 0.0
    dz = dataclasses.replace(dw, comp=c)
    B = MS.bases(dz, need=("A0", "T3"), min_pairs=MP)
    o = NF.run_n2(dz, B["T3"], MS.GROUPS, "LGBM", 0.0, 1, min_pairs=MP, min_closed=MC)
    k = int(np.flatnonzero(dz.el == d)[0])
    assert B["T3"]["fb"][d] == 1 and o["s2"][k] == B["A0"][d] and np.isfinite(o["s2"]).all() and (o["s2"] > 0).all()


def test_features_rows_up_to_d_unchanged_by_corruption_and_lead_mutant_caught():
    dw = synth()
    i0 = 300
    lb = np.log(dw.p.rv)
    X1, names = MS.feature_matrix(dw, lb)
    X2, _ = MS.feature_matrix(DC.corrupt(dw, i0), lb)
    assert X1.shape[1] == len(names) == 5 + 2 + 4 + 4 + 4
    np.testing.assert_array_equal(X1[:i0 + 1], X2[:i0 + 1])
    lead = lambda X: np.vstack([X[1:], np.full((1, X.shape[1]), np.nan)])         # МУТАНТ: признак строки t из t+1
    assert not np.array_equal(lead(X1)[:i0 + 1], lead(X2)[:i0 + 1], equal_nan=True)


def test_ext_rows_count_pairs_and_respect_segments():
    import types
    seg = np.r_[np.zeros(10, int), np.ones(10, int)]
    w = types.SimpleNamespace(p=types.SimpleNamespace(seg=seg, rv=np.ones(20)), el=np.array([12, 18]))
    des = np.ones((20, 2))
    des[6] = np.nan
    rows = MS.ext_rows(w, des, None, 4)
    assert rows[0] == 4 and 6 not in rows and rows.max() == 18
    assert 9 not in MS.ext_rows(w, des, None, 12) or MS.ext_rows(w, des, None, 12)[0] >= 10   # пары не склеиваются через границу сегмента


def test_spec_names_libs_separate_and_options():
    assert MLS.spec("N1x_A0")["lib"] == "XGB" and MLS.spec("N1l_A0")["lib"] == "LGBM"
    assert MLS.spec("N1x_B")["base"] == "H2" and MLS.spec("N2l")["base"] == "T3" and MLS.spec("N2l")["kind"] == "N2"
    a = MLS.spec("AblQ_x")
    assert a["groups"] == ("S", "T", "M") and a["parent"] == "N1x_A0" and a["base"] == "A0"
    c = MLS.spec("Cb_N1l_A0")
    assert c["mu"] == 0.05 and c["lib"] == "LGBM" and c["parent"] == "N1l_A0"
    assert all(MLS.spec(n)["lib"] in ("XGB", "LGBM") for n in ("N1x_A0", "N2x", "AblT_l"))


def test_diff_keys_exist_in_registry():
    if not (P.PRIOR_REP / "ALL_TRIALS.csv").exists():
        pytest.skip("нет реестра")
    ks = {(lib, c): MLS.diff_key(lib, c) for lib in ("XGB", "LGBM") for c in ("Ca", "Cc")}
    assert ks[("XGB", "Ca")] == "Ж-212:residual_qlike:e93f2dc703c2" and ks[("LGBM", "Cc")] == "Ж-212:residual_qlike:875dea70f204"
    assert set(ks.values()) <= set(L.load()["key"])
    assert L.LIMITS["structural_ml"] == 16


@pytest.mark.skipif(not (P.M10.exists() and P.COMP.exists()), reason="нет данных")
def test_bases_on_extended_rows_equal_stored_on_grid_real():
    import pandas as pd
    ctx = E.context()
    dw = MS.build_world(ctx["w"], pd.read_parquet(P.M10), pd.read_parquet(P.IMOEX))
    B = MS.bases(dw)
    chk = O.base_check(ctx, dw, B)
    assert max(chk.values()) <= 1e-12
    assert int(np.isfinite(B["A0"][:dw.el[0]]).sum()) >= 252 and int(np.isfinite(B["H2"][:dw.el[0]]).sum()) >= 252
    assert np.isfinite(B["A0"][dw.el]).all() and np.isfinite(B["H2"][dw.el]).all()
