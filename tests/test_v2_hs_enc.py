"""Ж-212: энкомпассинг (вопрос 8) на синтетике с известным ответом; OOS «по прошлому» — обрезка и мутация «закрыто + 1»; контроль знака."""
import numpy as np
import pytest

from models_v2.s8 import hs_enc, hs_pit


def synth(n=3000, gamma=0.0, seed=7):
    """base — прогноз-опора; δ — регрессор, истинная f = base·exp(γ·δ); RV = f·ε, E ε = 1."""
    rng = np.random.default_rng(seed)
    base = np.exp(rng.normal(size=n) * 0.4)
    delta = rng.normal(size=n) * 0.4
    rv = base * np.exp(gamma * delta) * rng.normal(size=n) ** 2
    d = (np.datetime64("2010-01-01", "D") + np.arange(n)).astype("datetime64[ns]")
    return rv, base, delta, d, d + np.timedelta64(1, "D")


def test_wald_no_information_not_significant_and_information_detected():
    rv, base, dl, *_ = synth(gamma=0.0)
    w0 = hs_enc.wald(rv, base, dl)
    assert abs(w0["t"][0]) < 3.0 and w0["p_joint"] > 0.003
    rv, base, dl, *_ = synth(gamma=1.0)
    w1 = hs_enc.wald(rv, base, dl)
    assert w1["gamma"][0] == pytest.approx(1.0, abs=0.15) and w1["p_joint"] < 1e-6 and w1["gain_in_sample"] > 0


def test_wald_sign_not_swapped():
    rv, base, dl, *_ = synth(gamma=1.0)
    assert hs_enc.wald(rv, base, dl)["gamma"][0] > 0.5 > hs_enc.wald(rv, base, -dl)["gamma"][0]      # мутация «перепутан знак» видна


def test_nw_cov_constant_series_is_zero_and_iid_matches_variance():
    assert np.allclose(hs_enc.nw_cov(np.ones((50, 2))), 0)
    x = np.random.default_rng(8).normal(size=(20000, 1))
    assert hs_enc.nw_cov(x)[0, 0] == pytest.approx(1.0, abs=0.08)


def test_mz_hac_known_slope():
    rng = np.random.default_rng(9)
    x = rng.normal(size=4000)
    r = hs_enc.mz_hac(0.3 + 2.0 * x + rng.normal(size=4000), x)
    assert r["b"] == pytest.approx(2.0, abs=0.1) and r["t_b"] > 20


def test_oos_gain_when_information_and_none_otherwise():
    rv, base, dl, d, td = synth(gamma=1.0)
    X = np.column_stack([np.ones(len(rv)), dl])
    s2, G = hs_enc.oos_expanding(rv, base, X, d, td, step=10)
    assert (G[:260] == 0).all() and np.allclose(s2[:252], base[:252])                    # до 252 закрытых — база
    assert hs_enc.dm_vs(rv, base, s2)["delta"] < 0 and hs_enc.dm_vs(rv, base, s2)["p"] < 0.01
    rv0, base0, dl0, d0, td0 = synth(gamma=0.0)
    s20, _ = hs_enc.oos_expanding(rv0, base0, np.column_stack([np.ones(len(rv0)), dl0]), d0, td0, step=10)
    assert hs_enc.dm_vs(rv0, base0, s20)["delta"] > -0.002                                # без информации «выигрыша» нет


def test_oos_truncation_and_mutation_closed_plus_one_caught():
    rv, base, dl, d, td = synth(n=800, gamma=1.0)
    X = np.column_stack([np.ones(len(rv)), dl])
    k = 600
    ref, _ = hs_enc.oos_expanding(rv, base, X, d, td, step=10)
    m = int(hs_pit.closed_count(d, td)[k])
    rv_b, X_b = rv.copy(), X.copy()
    rv_b[m:] = rv[m:] * 100.0                                        # цели, не закрытые на d_k, испорчены
    X_b[m:k, 1] += 3.0
    X_b[k + 1:, 1] += 3.0                                            # признаки после d_k испорчены; строка k сохранена
    got, _ = hs_enc.oos_expanding(rv_b, base, X_b, d, td, step=10)
    assert got[k] == ref[k]                                          # главный тест обрезки: прогноз на d_k не изменился
    mut, _ = hs_enc.oos_expanding(rv_b, base, X_b, d, td, step=10, extra=1)
    assert mut[k] != ref[k]                                          # мутация «закрыто + 1» видна


def test_oos_too_short_history_returns_base():
    rv, base, dl, d, td = synth(n=100, gamma=1.0)
    s2, G = hs_enc.oos_expanding(rv, base, np.column_stack([np.ones(100), dl]), d, td)
    assert np.array_equal(s2, base) and (G == 0).all()
