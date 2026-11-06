"""Ж-206, волна D, поток XGB: фолд LOYO с окном/весами (`d_fit.fold_train`), доступность окон, состав D1. Синтетика, без сети."""
import numpy as np
import pytest

from models_v2.s5 import pit
from models_v2.s5.b2 import feat, synth_b2
from models_v2.s6.xgb import d_fit, d_space

TR = np.concatenate([np.arange(0, 100), np.arange(200, 300)])
VAL = np.arange(100, 200)


def test_fold_window_takes_nearest_pairs_on_both_sides_and_never_val():
    out, w = d_fit.fold_train(TR, VAL, 40, 0, 400)
    assert w is None and len(out) == 40 and set(out) <= set(TR) and not set(out) & set(VAL)
    assert (np.diff(out) > 0).all()
    assert (out < 100).sum() == 20 and (out >= 200).sum() == 20                  # ближайшие с обеих сторон блока: 80…99 и 200…219


def test_fold_weights_decay_with_distance_to_validation_block():
    out, w = d_fit.fold_train(TR, VAL, 0, 100, 400)
    assert len(out) == len(TR) and w[99] == pytest.approx(0.5 ** (1 / 100)) and w[200] == pytest.approx(0.5 ** (1 / 100))
    assert w[0] < w[99] and w[299] < w[200] and w[150] == 0.0                    # вне обучения вес 0
    out0, w0 = d_fit.fold_train(TR, VAL, 0, 0, 400)
    assert w0 is None and (out0 == TR).all()                                      # expanding: ничего не меняется


def test_availability_counts_grow_and_share_is_zero_when_window_exceeds_history():
    s = synth_b2.World(*synth_b2.pair(700, 41))
    w = {"F": s.F, "dz0": s.dz0, "feats": s.feats, "acc": {g: list(c) for g, c in feat.GROUPS.items()}, "eps": pit.eps_floor(s.F.rv)}
    grid = np.arange(300, 600)
    n = d_space.pair_counts(s.F, d_space.spec_for(w), grid)
    assert (np.diff(n) >= 0).all() and n[-1] > n[0]
    av = d_space.availability(s.F, d_space.spec_for(w), grid)
    share = {k: v for (_, k, v) in av}
    assert share["roll1260"] == 0.0 and share["roll1008"] == 0.0               # на 700 синтетических барах окно 1008 недоступно ни разу


def test_d1_window_set_and_budget_limit():
    assert set(d_space.D1_WINDOWS) == {"roll504", "roll756", "roll1008", "roll1260", "exp756", "exp1260"}
    assert d_space.LIMIT == 30
