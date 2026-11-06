"""B4 (Ж-206, волна B, поток XGB): главный тест обрезки на каждой новой строке — прогноз на d побитно не меняется при порче/обрезке баров после 18:50 дня d.
Мутации: признаки +1 день (№ 1); пары с t+1 = d+1 (№ 15); нормировка весов L5 по всей выборке (будущее). Синтетика, без сети и реальных данных."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s5 import pit
from models_v2.s5.b1 import designs, jobs
from models_v2.s5.b2 import feat, synth_b2
from models_v2.s6.xgb import cfgx, worldx

D = 560
HP = synth_b2.HP
ACC = {g: list(c) for g, c in feat.GROUPS.items()}
CFGS = cfgx.NEW + cfgx.COND + [cfgx.mk("cond_obj", "qlike", 0.5, ("F4", "F2"), direction="feature")]


def as_world(s):
    return {"F": s.F, "dz0": s.dz0, "feats": s.feats, "acc": ACC, "eps": pit.eps_floor(s.F.rv)}


@pytest.fixture(scope="module")
def worlds():
    a, b, c = synth_b2.pair(700, 41)
    w = synth_b2.World(a, b, c)
    d = pd.Timestamp(w.F.dates[D])
    return w, synth_b2.World(synth_b2.damage(a, d), synth_b2.damage(b, d), synth_b2.damage(c, d))


def forecast(wd, c, T=D):
    spec = worldx.spec_of(wd, c)
    m = jobs.fit_at(wd["F"], spec, jobs.pairs(wd["F"], spec, T), HP)
    return np.asarray(m["predict"](spec.X[[D]]), float)


@pytest.mark.parametrize("c", CFGS, ids=lambda c: c["name"])
def test_main_truncation_test_every_new_row(worlds, c):
    w, dmg = worlds
    f1, f2 = forecast(as_world(w), c), forecast(as_world(dmg), c)
    assert np.isfinite(f1).all() and f1.tobytes() == f2.tobytes()


def test_truncation_test_is_not_vacuous_mutation_features_plus_one_day(worlds):
    w, dmg = worlds
    c = cfgx.NEW[5]                                                # f4_f2
    ww, dd = as_world(w), as_world(dmg)
    for x in (ww, dd):
        x["feats"] = x["feats"].shift(-1)                          # мутация № 1: признаки строки d видят d+1
    assert forecast(ww, c).tobytes() != forecast(dd, c).tobytes()


def test_mutation_pairs_with_target_d_plus_1_is_caught(worlds):
    w, dmg = worlds
    c = cfgx.NEW[1]                                                # x4_qlike_f4
    assert forecast(as_world(w), c, D + 1).tobytes() != forecast(as_world(dmg), c, D + 1).tobytes()


def test_mutation_weights_normalised_over_whole_sample_is_caught(worlds, monkeypatch):
    w, dmg = worlds
    c = cfgx.NEW[2]                                                # x5_l2_g05_f4

    def leaky(F, gw, idx):
        v = np.exp(-gw * F.X[:, 1])
        v = np.where(np.isfinite(v), v, 1.0)
        return v / v.mean()                                        # среднее по всем строкам, включая будущее
    monkeypatch.setattr(designs, "weights_l5", leaky)
    assert forecast(as_world(w), c).tobytes() != forecast(as_world(dmg), c).tobytes()


def test_weights_use_only_window_pairs(worlds):
    w, _ = worlds
    F = w.F
    idx = jobs.pairs(F, worldx.spec_of(as_world(w), cfgx.NEW[2]), D)
    wt = designs.weights_l5(F, 0.5, idx)
    assert wt[idx].mean() == pytest.approx(1.0, abs=1e-12) and idx.max() + 1 <= D
