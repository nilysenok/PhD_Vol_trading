"""Ж-211 (GATES_HF G1): R2/R3 и правила выбора по прошлому. λ=0 ≡ HAR-J, c=∞ ∧ λ=1 ≡ R1; clip ограничивает поправку; (c, λ) и веса на d не видят будущего;
мутации «закрыто+1», «клип без знака», «EMA на будущем веса», «лимит скорости нарушен» ловятся."""
import numpy as np
import pytest

from models_v2.s7 import hf_rules as R
from models_v2.s5.d import ens
from models_v2.s6.e import rules

N = 900


def synth(seed=5):
    rng = np.random.default_rng(seed)
    d = np.datetime64("2017-03-01") + np.arange(N).astype("timedelta64[D]")
    har = np.exp(rng.normal(0, 0.4, N))
    g = rng.normal(0, 0.2, N)
    g[::97] += 2.0
    rv = har * np.exp(0.5 * g + rng.normal(0, 0.3, N))
    return d.astype("datetime64[ns]"), (d + 1).astype("datetime64[ns]"), rv, har, g


def test_lambda_zero_is_har_bitwise():
    _, _, _, har, g = synth()
    assert R.apply(har, g, None, 0.0).tobytes() == har.tobytes()
    assert R.apply(har, g, 0.2, 0.0).tobytes() == har.tobytes()


def test_c_inf_lambda_one_is_r1():
    _, _, _, har, g = synth()
    assert np.allclose(R.apply(har, g, None, 1.0), har * np.exp(g), rtol=1e-15, atol=0)


@pytest.mark.parametrize("c", R.C_GRID)
def test_clip_bounds_correction_both_signs(c):
    _, _, _, har, g = synth()
    gg = np.log(R.apply(har, g, c, 1.0) / har)
    assert np.abs(gg).max() <= c + 1e-12
    assert (gg[g > c] > 0).all() and (gg[g < -c] < 0).all()      # знак сохраняется (мутация «без знака» красная)


def test_mutation_clip_without_sign_caught():
    _, _, _, har, g = synth()
    bad = har * np.exp(np.minimum(g, 0.2))                         # мутация: клип только сверху
    gg = np.log(bad / har)
    assert gg.max() <= 0.2 + 1e-12 and gg.min() < -0.2 - 1e-9          # нижняя граница не работает
    assert np.log(R.apply(har, g, 0.2, 1.0) / har).min() >= -0.2 - 1e-12


def perturb(rv, har, g, k0):
    rv, har, g = rv.copy(), har.copy(), g.copy()
    rv[k0:] *= 40.0
    g[k0:] *= -3.0
    return rv, har, g


def test_wf_select_ignores_unclosed_future_and_mutation_caught():
    d, td, rv, har, g = synth()
    cfgs = R.cfg_grid()
    k0 = 600
    rv2, har2, g2 = perturb(rv, har, g, k0)
    m = R.closed_of(rv, d, td)
    _, i1 = R.wf_select(rv, har, g, m, cfgs)
    _, i2 = R.wf_select(rv2, har2, g2, R.closed_of(rv2, d, td), cfgs)
    assert (i1[:k0 + 1] == i2[:k0 + 1]).all()
    mm = R.closed_of(rv, d, td, extra=100)                         # мутация: закрыто + 100 (видит цели строк k…k+99)
    mm2 = R.closed_of(rv2, d, td, extra=100)
    _, j1 = R.wf_select(rv, har, g, mm, cfgs)
    _, j2 = R.wf_select(rv2, har2, g2, mm2, cfgs)
    assert not (j1[:k0 + 1] == j2[:k0 + 1]).all()


def test_wf_default_is_har_before_min_closed():
    d, td, rv, har, g = synth()
    s2, idx = R.wf_select(rv, har, g, R.closed_of(rv, d, td), R.cfg_grid())
    k = int(np.argmax(R.closed_of(rv, d, td) >= R.MIN_CLOSED))
    assert (idx[:k] == 0).all() and s2[:k].tobytes() == har[:k].tobytes()


def test_cfg_grid_first_is_har_and_unique():
    cf = R.cfg_grid()
    assert cf[0] == (0.0, 1.0) and len(set(cf)) == len(cf) and (None, 1.0) in cf and (0.5, 0.25) in cf


def _targets(rv, d, td, S):
    return rules.weights("roll_126", S, {"rv": rv, "d": d, "td": td, "st_g4": np.zeros(N, bool)})


def test_ema_and_ratelimit_causal_and_bounded():
    d, td, rv, har, g = synth()
    S = np.column_stack([har, har * np.exp(g)])
    k0 = 500
    S2, rv2 = S.copy(), rv.copy()
    S2[k0:] *= 5.0
    rv2[k0:] *= 9.0
    W1, W2 = _targets(rv, d, td, S), _targets(rv2, d, td, S2)
    for f in (lambda W: R.ema_weights(W, 21), lambda W: R.rate_limit_weights(W, 0.02)):
        assert f(W1)[:k0 + 1].tobytes() == f(W2)[:k0 + 1].tobytes()
        assert np.allclose(f(W1).sum(axis=1), 1.0)
    w = R.rate_limit_weights(W1, 0.02)
    assert np.abs(np.diff(w, axis=0)).max() <= 0.02 + 1e-12


def test_mutation_ema_on_shifted_target_caught():
    d, td, rv, har, g = synth()
    S = np.column_stack([har, har * np.exp(g)])
    k0 = 500
    S2, rv2 = S.copy(), rv.copy()
    S2[k0:] *= 5.0
    rv2[k0:] *= 9.0
    W1, W2 = _targets(rv, d, td, S), _targets(rv2, d, td, S2)
    a = R.ema_weights(np.vstack([W1[100:], np.tile(W1[-1:], (100, 1))]), 5)   # мутация: вес d использует вес d+100 (закрыты и цели будущего)
    b = R.ema_weights(np.vstack([W2[100:], np.tile(W2[-1:], (100, 1))]), 5)
    assert (W1[k0 + 100] != W2[k0 + 100]).any()
    assert a[:k0 + 1].tobytes() != b[:k0 + 1].tobytes()


def test_closed_count_matches_ens():
    d, td, rv, _, _ = synth()
    assert (R.closed_of(rv, d, td) == ens.closed({"rv": rv, "d": d, "td": td})).all()
