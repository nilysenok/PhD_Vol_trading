"""G2 C: fitx ≡ fit_i1 при gamma = alpha = 0 и seed 42; оси gamma / alpha / seed действуют (мутанты MC4, MC6)."""
import numpy as np
import pytest

from models_v2.s4 import fit, runner
from models_v2.s5.b2 import engine, synth_b2
from models_v2.s5.c import fitx

T = 600


@pytest.fixture(scope="module")
def data():
    w = synth_b2.World(*synth_b2.pair(1000, 7, "2016-01-04"))
    dz = engine.make_dz(w.dz0, w.feats, engine.Cfg("x", add=("f4_park", "f4_range")))
    return w, dz, runner.pairs_upto(dz, w.F, T)


def pred(dz, idx, hp, seed=42):
    m = fitx.fit_i1x(dz.X, dz.y, idx, hp, None, None, seed)
    return m["predict"](dz.X[[T, T + 1, T + 2]]), m


def test_equals_fit_i1_bitwise(data):
    _, dz, idx = data
    hp = fitx.HPX(3, 10.0, 2.0, 0.1, 0.7, 0.6, 300)
    a, ma = pred(dz, idx, hp)
    mb = fit.fit_i1(dz.X, dz.y, idx, fit.HP(3, 10.0, 2.0, 0.1, 0.7, 0.6, 300))
    assert a.tobytes() == mb["predict"](dz.X[[T, T + 1, T + 2]]).tobytes() and ma["k"] == mb["k"] and ma["s2_es"] == mb["s2_es"]


def test_gamma_alpha_seed_act(data):
    _, dz, idx = data
    base = fitx.HPX(3, 10.0, 2.0, 0.1, 0.7, 0.6, 300)
    p0, _ = pred(dz, idx, base)
    for hp in (fitx.HPX(3, 10.0, 2.0, 0.1, 0.7, 0.6, 300, 1.0, 0.0), fitx.HPX(3, 10.0, 2.0, 0.1, 0.7, 0.6, 300, 0.0, 5.0)):   # проводка осей, не диапазон
        assert pred(dz, idx, hp)[0].tobytes() != p0.tobytes()     # мутант MC4: ось игнорируется
    assert pred(dz, idx, base, 43)[0].tobytes() != p0.tobytes()   # мутант MC6: seed не доходит до модели
