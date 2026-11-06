"""G2 (Ж-206, волна D, поток XGB): главный тест обрезки на каждой новой строке — прогноз на d побитно не меняется при порче/обрезке баров после 18:50 дня d.
Строки: rolling 504/756/1008/1260, веса 756/1260, модель блока (обучена на T < d: еженедельно / ежемесячно), режимные признаки R1 (qw / all), состояние R4. Синтетика.
Мутации: признаки +1 день (№ 1) и режимные признаки +1 день краснят тест."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s5 import pit
from models_v2.s5.b1 import designs, jobs
from models_v2.s5.b2 import feat, synth_b2
from models_v2.s6.xgb import c_space, d_comp, d_fit, d_space

D = 560
ACC = {g: list(c) for g, c in feat.GROUPS.items()}
HP = c_space.to_hpc(synth_b2.HP)
ROWS = [("roll504", ""), ("roll756", ""), ("roll1008", ""), ("roll1260", ""), ("exp756", ""), ("exp1260", ""), ("expanding", "qw"), ("expanding", "all")]


def as_world(s):
    return {"F": s.F, "dz0": s.dz0, "feats": s.feats, "acc": ACC, "eps": pit.eps_floor(s.F.rv)}


@pytest.fixture(scope="module")
def worlds():
    a, b, c = synth_b2.pair(700, 41)
    w = synth_b2.World(a, b, c)
    d = pd.Timestamp(w.F.dates[D])
    return w, synth_b2.World(synth_b2.damage(a, d), synth_b2.damage(b, d), synth_b2.damage(c, d))


def forecast(wd, win, r1, T=D):
    spec = d_space.spec_for(wd, r1)
    n, hl = d_space.WINDOWS[win]
    m = d_fit.fit_d(spec.X, spec.y, jobs.pairs(wd["F"], spec, T), HP, n, hl, T)
    return np.asarray(m["predict"](spec.X[[D]]), float)


@pytest.mark.parametrize("win,r1", ROWS, ids=lambda v: str(v))
def test_main_truncation_test_every_new_row(worlds, win, r1):
    w, dmg = worlds
    f1, f2 = forecast(as_world(w), win, r1), forecast(as_world(dmg), win, r1)
    assert np.isfinite(f1).all() and f1.tobytes() == f2.tobytes()


@pytest.mark.parametrize("win", ["roll504", "exp756", "expanding"])
def test_block_model_trained_before_d_truncation(worlds, win):
    """Еженедельный/ежемесячный блок: модель обучена на T = D − 5 (первая дата блока), прогноз на D — по X[D]; порча после D не меняет."""
    w, dmg = worlds
    assert forecast(as_world(w), win, "", D - 5).tobytes() == forecast(as_world(dmg), win, "", D - 5).tobytes()


def test_state_r4_truncation(worlds):
    w, dmg = worlds
    a, b = d_comp.stress_state(w.F), d_comp.stress_state(dmg.F)
    assert a[:D + 1].tobytes() == b[:D + 1].tobytes()


@pytest.mark.parametrize("win,r1", [("expanding", "all"), ("expanding", "qw")])
def test_truncation_test_is_not_vacuous_mutation_regime_plus_one_day(worlds, monkeypatch, win, r1):
    """Мутация: режимные столбцы строки t берутся со строки t+1 — прогноз на D начинает зависеть от данных после D."""
    w, dmg = worlds
    real = designs.regime_cols

    def shifted(F):
        return {k: np.concatenate([v[1:], [np.nan]]) for k, v in real(F).items()}
    monkeypatch.setattr(designs, "regime_cols", shifted)
    assert forecast(as_world(w), win, r1).tobytes() != forecast(as_world(dmg), win, r1).tobytes()


def test_truncation_test_is_not_vacuous_mutation_features_plus_one_day(worlds):
    w, dmg = worlds
    ww, dd = as_world(w), as_world(dmg)
    for x in (ww, dd):
        x["feats"] = x["feats"].shift(-1)
    assert forecast(ww, "roll504", "").tobytes() != forecast(dd, "roll504", "").tobytes()
