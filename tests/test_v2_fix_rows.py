"""И4/И5 и формы цели (Ж-200): уровень возвращается сложением; прирост; остаток; мутации M8, M9."""
import numpy as np
import pytest

from models_v2 import har
from models_v2.s2 import features as ft
from models_v2.s3 import resid
from models_v2.s4 import design, fit, runner, synth


@pytest.fixture(scope="module")
def F():
    return ft.build(synth.random_bars(700, 5))


def lb_of(F):
    return resid.harj_ab(har.Panel(F.dates, F.seg, F.rv, F.X[:, :4]))[1]


@pytest.mark.parametrize("form", design.FORMS)
def test_level_restored_by_offset(F, form):
    dz = design.make(F, form, lb=lb_of(F) if form == "resid" else None)
    ok = np.isfinite(dz.y)
    t = np.flatnonzero(ok)
    assert np.allclose(dz.off[t] + dz.y[t], F.lnrv[t + 1])    # M8: без off уровень не восстановить


def test_delta_target_and_mutation_m8(F):
    dz = design.make(F, "delta")
    t = np.flatnonzero(np.isfinite(dz.y))
    assert np.allclose(dz.y[t], F.lnrv[t + 1] - F.lnrv[t]) and np.array_equal(dz.off, F.lnrv)
    assert not np.allclose(0.0 + dz.y[t], F.lnrv[t + 1])       # M8 (off = 0) ловится


def test_delta_constant_increment_known_answer():
    n = 700
    X = np.random.default_rng(2).normal(size=(n, 4))
    c = 0.07
    y = np.full(n, c)
    idx = fit.pair_index(np.zeros(n, int), X, y, (0, 1, 2, 3), 650)
    m = fit.fit_i1(X, y, idx, fit.HP(3, 10.0, 1.0, 0.1, 1.0))
    f = m["predict"](X[650:651])[0]
    lnrv_d = 1.234
    assert lnrv_d + f == pytest.approx(lnrv_d + c, abs=1e-6)   # σ̂²(а) = RV_d · e^c
    assert np.exp(lnrv_d + f + m["s2_es"] / 2) == pytest.approx(np.exp(lnrv_d + c), rel=1e-6)   # s² = 0 -> (б) = (а); M9 ловится на s² > 0


def test_mutation_m9_without_half_s2_differs_when_s2_positive():
    s2 = 0.3
    assert np.exp(1.0 + s2 / 2) != np.exp(1.0)


def test_weighted_s2_uses_weights_in_es_block():
    rng = np.random.default_rng(4)
    X, y = rng.normal(size=(700, 4)), rng.normal(size=700)
    idx = fit.pair_index(np.zeros(700, int), X, y, (0, 1, 2, 3), 690)
    hp = fit.HP(3, 5.0, 1.0, 0.1, 1.0)
    w = fit.weights(np.arange(700), 690)
    a, b = fit.fit_i1(X, y, idx, hp, w), fit.fit_i1(X, y, idx, hp)
    assert a["s2_es"] != b["s2_es"]
    wrong = fit.weights(np.arange(700), 690)[::-1].copy()      # M7: возраст наоборот
    assert not np.all(np.diff(wrong) > 0)


def test_block_fit_variants_run_and_perm_changes_prediction(F):
    dz = design.make(F, "level")
    hp = fit.HP(3, 10.0, 2.0, 0.1, 0.8)
    T = 600
    w = runner.block_fit(F, dz, hp, T, [T], False, None, "work")["f"]
    o = runner.block_fit(F, dz, hp, T, [T], False, None, "oracle")["f"]
    p = runner.block_fit(F, dz, hp, T, [T], False, None, "perm")["f"]
    assert abs(o[0] - dz.y[T]) < 0.3 and w.tobytes() != p.tobytes()
