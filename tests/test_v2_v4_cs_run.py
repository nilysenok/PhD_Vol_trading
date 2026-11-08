"""V4-CS (G27): CS1/CS2 — PIT боевого пути `v4_cs_run`, расписание, тождество с C8 при n → 0, мутации «закрыто + 1» и «κ_i / λ от будущих пар», LOTO, известный ответ оценщика."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s11 import v4_b_run as BR, v4_cb_fit as FB, v4_cb_run as CB, v4_cs_fit as FS, v4_cs_run as R, v4_z_feat as ZF
from tests.test_v2_v4_z_feat import synth


@pytest.fixture(scope="module")
def data():
    return synth(560)


def day_sep(P):
    d = P["date"]
    return d[d.dt.to_period("M") == pd.Period("2017-09")].min()


def test_schedule_known_values():
    assert FS.sched(0) == 250.0 and FS.sched(250) == 125.0 and FS.sched(750) == 62.5
    k, lam = FS.penalties("CS1", 750, [10, 20])
    assert list(k) == [250.0, 250.0] and lam == 62.5
    k, lam = FS.penalties("CS2", 750, [0, 250, 750])
    assert list(k) == [250.0, 125.0, 62.5] and lam == 250.0
    with pytest.raises(ValueError):
        FS.penalties("C8", 1, [1])


def _syn_fit(n=3, N=900, seed=3):
    rng = np.random.default_rng(seed)
    tid = rng.integers(0, n, N)
    ell = rng.normal(0, 0.3, N)
    lH = rng.normal(0, 0.3, N)
    z = rng.normal(0, 1, N)
    rv = np.exp(lH + 0.7 * ell + rng.normal(0, 0.3, N))
    return rv, lH, ell, z, tid, n


def test_vector_fit_with_constant_penalty_equals_c8_fit_1e8():
    rv, lH, ell, z, tid, n = _syn_fit()
    a = FB.fit_regime(rv, lH, ell, z, tid, n)
    b = FS.fit_regime_vec(rv, lH, ell, z, tid, n, np.full(n, 250.0), 250.0)
    for key in ("a", "b", "a_calm", "a_crisis"):
        assert abs(a[key] - b[key]) < 1e-8, key
    assert np.allclose(a["d_calm"], b["d_calm"], atol=1e-8) and np.allclose(a["d_crisis"], b["d_crisis"], atol=1e-8)


def test_smaller_kappa_lets_ticker_deviation_grow_known_direction():
    rv, lH, ell, z, tid, n = _syn_fit()
    rv = np.where(tid == 0, rv * np.exp(0.8 * np.abs(ell)), rv)          # бумага 0 отличается от панели
    strong = FS.fit_regime_vec(rv, lH, ell, z, tid, n, np.full(n, 250.0), 250.0)
    weak = FS.fit_regime_vec(rv, lH, ell, z, tid, n, np.full(n, 2.0), 250.0)
    assert np.abs(weak["d_calm"]).sum() + np.abs(weak["d_crisis"]).sum() > np.abs(strong["d_calm"]).sum() + np.abs(strong["d_crisis"]).sum()


@pytest.mark.parametrize("cfg", R.CFGS)
def test_identity_with_c8_when_schedule_is_flat_n_zero(data, monkeypatch, cfg):
    """При n → 0 расписание равно 250 (C8): CS1 и CS2 с sched ≡ 250 воспроизводят C8 (допуск 1e-8) на всех строках."""
    P, inp = data
    zl = ZF.build_panel_z(P, inp)
    ref = CB.forecast("C8", P, zl)[0]["F"].to_numpy()
    monkeypatch.setattr(FS, "sched", lambda n: np.full(np.shape(n), 250.0) if np.ndim(n) else 250.0)
    assert np.allclose(R.forecast(cfg, P, zl)[0]["F"].to_numpy(), ref, rtol=1e-8, atol=0)


@pytest.mark.parametrize("cfg", R.CFGS)
def test_prior_before_60_pairs_is_c8_and_later_months_differ(data, cfg):
    P, inp = data
    zl = ZF.build_panel_z(P, inp)
    a, b = R.forecast(cfg, P, zl)[0], CB.forecast("C8", P, zl)[0]
    jan = (a["date"] < "2017-02-01").to_numpy()
    assert np.array_equal(a["F"].to_numpy()[jan], b["F"].to_numpy()[jan]) and np.allclose(a["w"][jan], 0.5)
    assert not np.allclose(a["F"].to_numpy()[~jan], b["F"].to_numpy()[~jan], rtol=1e-6)       # затухание штрафа реально меняет прогноз


@pytest.mark.parametrize("cfg", R.CFGS)
def test_cut_future_data_does_not_change_forecast(data, cfg):
    P, inp = data
    assert not R.pit_violation(cfg, P, inp, day_sep(P))


@pytest.mark.parametrize("cfg", R.CFGS)
def test_mutation_closed_plus_one_is_caught_via_real_path(data, monkeypatch, cfg):
    P, inp = data
    monkeypatch.setattr(BR, "closed_mask", lambda t_plus, d0: np.asarray(t_plus) <= np.datetime64(d0) + np.timedelta64(3, "D"))
    assert R.pit_violation(cfg, P, inp, day_sep(P))


@pytest.mark.parametrize("cfg", R.CFGS)
def test_mutation_penalty_from_future_pairs_is_caught(data, monkeypatch, cfg):
    """κ_i (CS2) / λ (CS1) от числа ВСЕХ пар бумаги, включая будущие, ловится обрезкой на боевом пути."""
    P, inp = data
    monkeypatch.setattr(R, "count_pairs", lambda closed, tix, nT: np.bincount(tix, minlength=nT))
    assert R.pit_violation(cfg, P, inp, day_sep(P))


def test_cs2_kappa_i_decreases_with_ticker_pairs_in_weights(data):
    P, inp = data
    _, w = R.forecast("CS2", P, ZF.build_panel_z(P, inp))
    t = w[(w["ticker"] == "AFLT") & (w["n_pairs"] > 0)].sort_values("month")
    assert (np.diff(t["kappa_i"].to_numpy()) < 0).all() and np.allclose(t["kappa_i"], 250.0 * 250.0 / (250.0 + t["n_pairs"]))
    pn = w[w["ticker"] == "PANEL"]
    assert (pn["lam"] == 250.0).all() and pn["converged"].mean() > 0.9


def test_cs1_lambda_decreases_with_panel_pairs_in_weights(data):
    P, inp = data
    _, w = R.forecast("CS1", P, ZF.build_panel_z(P, inp))
    pn = w[(w["ticker"] == "PANEL") & (w["n_pairs"] >= 60)].sort_values("month")
    assert np.allclose(pn["lam"], 250.0 * 250.0 / (250.0 + pn["n_pairs"])) and (np.diff(pn["lam"].to_numpy()) < 0).all()


def test_loto_excluded_ticker_has_panel_parameters_and_does_not_enter_cross_or_training(data):
    P, inp = data
    held, others = "IRAO", ["AFLT", "ALRS", "HYDR"]
    zl = ZF.build_panel_z(P, inp)
    P2 = P.copy()
    P2.loc[P2["ticker"] == held, "rv"] *= 7.3
    inp2 = {t: dict(v) for t, v in inp.items()}
    inp2[held] = {"daily": inp[held]["daily"].assign(rv=inp[held]["daily"]["rv"] * 7.3), "herf": inp[held]["herf"]}
    zl2 = ZF.build_panel_z(P2, inp2)
    for cfg in R.CFGS:
        a, wa = R.forecast(cfg, P, ZF.recross(zl, others), exclude=held)
        b, _ = R.forecast(cfg, P2, ZF.recross(zl2, others), exclude=held)
        assert np.array_equal(a[a["ticker"].isin(others)]["F"].to_numpy(), b[b["ticker"].isin(others)]["F"].to_numpy()), cfg
        h = wa[wa["ticker"] == held]
        assert (h["dev"] == 0).all() and (h["n_pairs"] == 0).all() and h["excluded"].all(), cfg
