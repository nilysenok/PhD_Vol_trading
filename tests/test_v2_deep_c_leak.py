"""G6 C: главный тест обрезки на модельных строках C (HP с gamma/alpha/colsample < 1, seed 43); мутант «признаки +1 день» (MC5, № 1)."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s5.b2 import engine, feat, synth_b2
from models_v2.s5.c import engine_c, fitx

D = 560
HP = fitx.HPX(2, 20.0, 5.0, 0.1, 0.7, 0.6, 300, 0.01, 1.0)
ADD = {"f4": feat.GROUPS["F4"], "f4_range": ("f4_range",), "f4_park": ("f4_park",), "f4_f1": feat.GROUPS["F4"] + feat.GROUPS["F1"],
       "top2": feat.GROUPS["F4"] + feat.GROUPS["F3"]}


@pytest.fixture(scope="module")
def worlds():
    a, b, c = synth_b2.pair(700, 41)
    w = synth_b2.World(a, b, c)
    d = pd.Timestamp(w.F.dates[D])
    return w, synth_b2.World(synth_b2.damage(a, d), synth_b2.damage(b, d), synth_b2.damage(c, d))


def f_at(w, name, seed=43):
    dz = engine.make_dz(w.dz0, w.feats, engine.Cfg("c_" + name, add=ADD[name]))
    engine_c._init(engine_c.W(w.F, np.array([D]), {name: dz}))
    return engine_c._day_job((name, D, HP, seed))[1]["f"]


@pytest.mark.parametrize("name", list(ADD))
def test_main_truncation_test_every_c1_row(worlds, name):
    w, dmg = worlds
    assert np.float64(f_at(w, name)).tobytes() == np.float64(f_at(dmg, name)).tobytes()


def test_mutation_features_plus_one_day_caught(worlds):
    w, dmg = worlds
    for x in (w, dmg):
        x.feats = x.feats.shift(-1)
        x.dz0 = type(x.dz0)(np.vstack([x.dz0.X[1:], x.dz0.X[-1:]]), x.dz0.y, x.dz0.off)
    try:
        assert f_at(w, "f4") != f_at(dmg, "f4")
    finally:
        for x in (w, dmg):
            x.feats = x.feats.shift(1)
