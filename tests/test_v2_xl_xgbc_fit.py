"""C2 (Ж-206, волна C, поток XGB): обучение c_fit — нулевой сдвиг побитно = fit.fit_i1 (l2) и fitobj (gamma); каждая ось (gamma, reg_alpha, max_delta_step, seed) доходит до бустера.
Мутации: ось выкинута из параметров (тест на ось падает бы). Синтетика, без сети и реальных данных."""
import dataclasses

import numpy as np
import pytest

from models_v2.s4 import fit
from models_v2.s5 import pit
from models_v2.s5.b1 import fitobj, jobs
from models_v2.s5.b2 import feat, synth_b2
from models_v2.s6.xgb import c_calc, c_fit, c_space, worldx

D = 560
ACC = {g: list(c) for g, c in feat.GROUPS.items()}
BASE = c_space.to_hpc(synth_b2.HP)
STRONG = {"gamma": dict(gamma=5.0), "alpha": dict(alpha=5.0), "mds": dict(mds=0.001), "seed": None,
          "depth": dict(depth=1), "min_child": dict(min_child=200.0), "lr": dict(lr=0.02), "subsample": dict(subsample=0.5),
          "colsample": dict(colsample=0.3), "l2": dict(l2=100.0)}


@pytest.fixture(scope="module")
def world():
    s = synth_b2.World(*synth_b2.pair(700, 41))
    return {"F": s.F, "dz0": s.dz0, "feats": s.feats, "acc": ACC, "eps": pit.eps_floor(s.F.rv)}


def run(world, loss, hp, seed=fit.SEED):
    spec = worldx.spec_of(world, c_calc.SPEC_CFG[loss])
    idx = jobs.pairs(world["F"], spec, D)
    return spec, idx, c_fit.fit_c(spec.X, spec.y, idx, hp, loss, seed)


def f_at(world, loss, hp, seed=fit.SEED):
    spec, _, m = run(world, loss, hp, seed)
    return float(m["predict"](spec.X[[D]])[0]), m["k"]


def test_neutral_l2_bitwise_equals_fit_i1(world):
    spec, idx, m = run(world, "l2", BASE)
    ref = fit.fit_i1(spec.X, spec.y, idx, synth_b2.HP)
    assert m["predict"](spec.X[[D]]).tobytes() == ref["predict"](spec.X[[D]]).tobytes()
    assert m["k"] == ref["k"] and np.float64(m["s2_es"]).tobytes() == np.float64(ref["s2_es"]).tobytes()


def test_neutral_gamma_bitwise_equals_fitobj(world):
    spec, idx, m = run(world, "gamma", BASE)
    ref = fitobj.fit_loss(spec.X, spec.y, idx, synth_b2.HP, "gamma", None)
    assert m["predict"](spec.X[[D]]).tobytes() == ref["predict"](spec.X[[D]]).tobytes() and m["k"] == ref["k"]


@pytest.mark.parametrize("loss", ["l2", "gamma"])
@pytest.mark.parametrize("axis", [a for a in STRONG if a != "seed"])
def test_every_axis_reaches_booster(world, loss, axis):
    hp = dataclasses.replace(BASE, **STRONG[axis])
    assert f_at(world, loss, hp) != f_at(world, loss, BASE)


@pytest.mark.parametrize("loss", ["l2", "gamma"])
def test_seed_changes_forecast_when_subsample_below_one(world, loss):
    assert f_at(world, loss, BASE, 43)[0] != f_at(world, loss, BASE, 42)[0]
    assert f_at(world, loss, BASE, 43) == f_at(world, loss, BASE, 43)       # детерминирован


def test_mds_zero_is_off_and_not_written():
    assert "max_delta_step" not in c_fit.params(BASE, 0.1, 42, "l2")
    assert c_fit.params(dataclasses.replace(BASE, mds=0.3), 0.1, 42, "l2")["max_delta_step"] == 0.3


@pytest.mark.parametrize("key,axis", [("gamma", "gamma"), ("reg_alpha", "alpha"), ("max_delta_step", "mds")])
def test_mutation_axis_dropped_from_params_is_caught(world, monkeypatch, key, axis):
    real = c_fit.params

    def dropped(hp, base, seed, loss):
        p = real(hp, base, seed, loss)
        p.pop(key, None)
        return p
    monkeypatch.setattr(c_fit, "params", dropped)
    hp = dataclasses.replace(BASE, **STRONG[axis])
    assert f_at(world, "l2", hp) == f_at(world, "l2", BASE)                 # мутант не отличим от опоры: тест every_axis на нём краснеет


def test_mutation_seed_not_passed_is_caught(world, monkeypatch):
    real = c_fit.params
    monkeypatch.setattr(c_fit, "params", lambda hp, base, seed, loss: real(hp, base, fit.SEED, loss))
    assert f_at(world, "l2", BASE, 43) == f_at(world, "l2", BASE, 42)


def test_gamma_family_finite_positive_forecast(world):
    spec, _, m = run(world, "gamma", BASE)
    assert np.isfinite(m["predict"](spec.X[[D]])).all() and np.isnan(m["s2_es"])
