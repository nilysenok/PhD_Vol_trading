"""G3 C: границы Optuna, J только по Y ≤ 2021, PIT-выбор HP_Y, фолд LOYO без утечки, детерминизм, бюджет (MC1–MC3, MC15)."""
import numpy as np
import optuna
import pytest

from models_v2.s4 import fit, select
from models_v2.s5.b2 import engine, loyo, synth_b2
from models_v2.s5.c import engine_c, fitx, space, tab_c, tpe

HP = fitx.HPX(3, 10.0, 2.0, 0.1, 0.8, 1.0, 300)


def study(seed=42, n=12):
    s = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=4))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    hps = []

    def obj(t):
        hp = space.suggest(t)
        hps.append(hp)
        return hp.depth + hp.lr + hp.gamma
    s.optimize(obj, n_trials=n)
    return hps


def test_space_bounds_and_determinism():
    a, b = study(), study()
    assert a == b
    for h in a:
        assert 1 <= h.depth <= 3 and 10 <= h.min_child <= 200 and 0.5 <= h.l2 <= 100 and 0.02 <= h.lr <= 0.1
        assert 0.5 <= h.subsample <= 1 and 0.3 <= h.colsample <= 1 and 0 <= h.gamma <= 0.05 and 0 <= h.alpha <= 5 and h.max_trees == 300
    assert space.N_TPE == 22 and space.GUIDE_YEARS == (2017, 2018, 2019, 2020, 2021)


@pytest.fixture(scope="module")
def world():
    return synth_b2.World(*synth_b2.pair(1000, 7, "2016-01-04"))


def sc_of(w, dz, years, hp=HP):
    return {Y: float(np.nanmean([engine_c.fold(w.F, dz, hp, select.cut_row(w.F, Y), y) for y in range(2015, Y)])) for Y in years}


def test_J_ignores_targets_from_guide_cut_and_mutant_caught(world):
    w = world
    dz = engine.make_dz(w.dz0, w.feats, engine.Cfg("x"))
    bad = type(dz)(dz.X, dz.y.copy(), dz.off)
    bad.y[w.F.dates >= np.datetime64("2018-12-31")] += 5.0    # цели ≥ 01.01.2019 (y[t] — цель на t+1)
    gy = (2018, 2019, 2020)
    assert tpe.J(sc_of(w, dz, gy), gy[:2]) == tpe.J(sc_of(w, bad, gy), gy[:2])    # Y ≤ 2018 видит цели < 01.01.2018/2019
    assert tpe.J(sc_of(w, dz, gy), gy) != tpe.J(sc_of(w, bad, gy), gy)            # мутант MC3: Y = 2019 тоже в J (после cut) — меняется


def test_fold_equals_loyo_fold_and_mutant_cut_plus_one(world):
    w = world
    dz = engine.make_dz(w.dz0, w.feats, engine.Cfg("x"))
    Tc = select.cut_row(w.F, 2018)
    assert engine_c.fold(w.F, dz, HP, Tc, 2017) == loyo.fold(w.F, dz, engine.Cfg("x"), fit.HP(3, 10.0, 2.0, 0.1, 0.8, 1.0, 300), Tc, 2017)
    assert engine_c.fold(w.F, dz, HP, Tc + 1, 2017) != engine_c.fold(w.F, dz, HP, Tc, 2017)   # MC1: окно за cut — другой результат


def test_choice_uses_only_score_Y():
    hps = {k: fitx.HPX(k + 1, 30.0, 2.0, 0.05, 0.8, 1.0, 300) for k in range(3)}
    i2 = {Y: (fit.HP(2, 30.0, 2.0, 0.05, 0.8, 0.5, 300), 1.0) for Y in range(2017, 2020)}
    sc = lambda v: {Y: v[Y - 2017] for Y in range(2017, 2020)}   # noqa: E731
    cand = {0: (hps[0], sc([0.5, 0.9, 0.9])), 1: (hps[1], sc([0.9, 0.4, 0.9])), 2: (hps[2], sc([0.9, 0.9, 0.3]))}
    assert tpe.choose(cand, i2, 2018)[0] == hps[1]
    other = {k: (h, sc([v if Y == 2018 else 0.01 for Y, v in zip(range(2017, 2020), [0.9, 0.4, 0.9])])) for k, (h, _) in cand.items() if k == 1}
    cand2 = dict(cand)
    cand2[2] = (hps[2], sc([0.01, 0.9, 0.01]))      # меняем только другие годы
    assert tpe.choose(cand2, i2, 2018)[0] == hps[1] and other
    assert tpe.choose(cand, i2, 2018)[2] == 0.4


def row(dirn):
    return {"направление": dirn}


def test_budget_guard_and_unique_ids():
    tab_c.check_budget([row("feature_engineering")] * 25, {"feature_engineering": 12})
    with pytest.raises(SystemExit):
        tab_c.check_budget([row("feature_engineering")] * 26, {"feature_engineering": 12})
    with pytest.raises(SystemExit):
        tab_c.check_budget([row("regime_aware")] * 10, {"regime_aware": 45})
    ids = {space.trial_id({"k": i, "hp": space.hp_dict(h)}) for i, h in enumerate(study())}
    assert len(ids) == 12
