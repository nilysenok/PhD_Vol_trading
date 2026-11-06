"""C3 (Ж-206, волна C, поток XGB): главный тест обрезки на каждой новой строке — прогноз на d побитно не меняется при порче/обрезке баров после 18:50 дня d.
Строки: l2 и gamma × 9 осей (сдвиг HP) × seed. Мутация: признаки +1 день — тест краснеет. Синтетика."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from models_v2.s5 import pit
from models_v2.s5.b1 import jobs
from models_v2.s5.b2 import feat, synth_b2
from models_v2.s6.xgb import c_calc, c_fit, c_space, worldx

D = 560
ACC = {g: list(c) for g, c in feat.GROUPS.items()}
BASE = {Y: c_space.to_hpc(synth_b2.HP) for Y in range(2017, 2027)}
POINTS = [(ax, u) for ax, spec in c_space.AXES.items() for u in spec[4]]


def as_world(s):
    return {"F": s.F, "dz0": s.dz0, "feats": s.feats, "acc": ACC, "eps": pit.eps_floor(s.F.rv)}


@pytest.fixture(scope="module")
def worlds():
    a, b, c = synth_b2.pair(700, 41)
    w = synth_b2.World(a, b, c)
    d = pd.Timestamp(w.F.dates[D])
    return w, synth_b2.World(synth_b2.damage(a, d), synth_b2.damage(b, d), synth_b2.damage(c, d))


def forecast(wd, family, hp, seed=42):
    spec = worldx.spec_of(wd, c_calc.SPEC_CFG[family])
    m = c_fit.fit_c(spec.X, spec.y, jobs.pairs(wd["F"], spec, D), hp, family, seed)
    return np.asarray(m["predict"](spec.X[[D]]), float)


@pytest.mark.parametrize("family", ["l2", "gamma"])
@pytest.mark.parametrize("axis,u", POINTS, ids=lambda v: str(v))
def test_main_truncation_test_every_new_row(worlds, family, axis, u):
    w, dmg = worlds
    hp = c_space.apply(BASE, {axis: u})[2020]
    f1, f2 = forecast(as_world(w), family, hp), forecast(as_world(dmg), family, hp)
    assert np.isfinite(f1).all() and f1.tobytes() == f2.tobytes()


@pytest.mark.parametrize("family", ["l2", "gamma"])
def test_seed_row_truncation(worlds, family):
    w, dmg = worlds
    hp = BASE[2020]
    assert forecast(as_world(w), family, hp, 45).tobytes() == forecast(as_world(dmg), family, hp, 45).tobytes()


@pytest.mark.parametrize("family", ["l2", "gamma"])
def test_truncation_test_is_not_vacuous_mutation_features_plus_one_day(worlds, family):
    w, dmg = worlds
    ww, dd = as_world(w), as_world(dmg)
    for x in (ww, dd):
        x["feats"] = x["feats"].shift(-1)                          # мутация № 1: признаки строки d видят d+1
    hp = dataclasses.replace(BASE[2020], gamma=0.0)
    assert forecast(ww, family, hp).tobytes() != forecast(dd, family, hp).tobytes()
