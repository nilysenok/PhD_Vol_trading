"""Ж-214, лист 2, V5 (QLIKE-HAR-J): известный ответ QLIKE-подгонки, ограничения β >= 0 (мутант без ограничений ловится), положительность,
выбор h и шкала c по закрытым прогнозам (мутант «закрыто + 1» ловится), главный тест обрезки на каждой новой строке (Q1, Q2, Q4, Q5),
мутант «пары t+1 = d+1» ловится тестом обрезки. Синтетика, без данных и сети."""
import numpy as np
import pandas as pd
import pytest

from models_v2 import har as H
from models_v2.s9 import hv2_eval as E, hv2_qlev as QL, hv2_qlog as QG, hv2_world as W


def synth_world(n=320, seed=3, tail=40):
    rng = np.random.default_rng(seed)
    ln = np.zeros(n)
    for t in range(1, n):
        ln[t] = 0.8 * ln[t - 1] + rng.normal(0, 0.5)
    rv = np.exp(ln + 1.0)
    s = pd.Series(rv)
    j = np.abs(rng.normal(0, 0.3, n)) * (rng.random(n) < 0.4)
    X = np.column_stack([np.log(rv), np.log(s.rolling(5, min_periods=1).mean()), np.log(s.rolling(22, min_periods=1).mean()), np.log1p(j)])
    p = H.Panel(pd.bdate_range("2018-01-01", periods=n).to_numpy("datetime64[ns]"), np.zeros(n, int), rv, X)
    rq = rv ** 2 * np.exp(rng.normal(0, 0.3, n))
    w = W.World(p, rq, np.sqrt(2 * rq / 50) / rv, H.eligible(p))
    return W.World(p, w.rq, w.rel, w.el[-tail:])


def test_fit_levels_recovers_known_answer_noiseless():
    rng = np.random.default_rng(1)
    Z = np.column_stack([np.ones(300), rng.uniform(1, 5, 300), rng.uniform(1, 5, 300), rng.uniform(1, 5, 300), rng.uniform(0, 1, 300)])
    b = np.array([0.5, 0.3, 0.2, 0.1, 0.0])
    got = QL.fit_levels(Z, np.einsum("ij,j->i", Z, b))
    assert np.allclose(got, b, atol=2e-4), got


def test_nonneg_active_and_unconstrained_mutant_caught():
    rng = np.random.default_rng(2)
    Z = np.column_stack([np.ones(400), rng.uniform(2, 6, 400), rng.uniform(2, 6, 400)])
    y = np.einsum("ij,j->i", Z, np.array([2.0, 0.8, -0.3]))                     # истинный β_w < 0, но f > 0
    ok = QL.fit_levels(Z, y)
    assert ok.min() >= 0 and ok[2] <= 1e-12 and ok[0] >= QL.FLOOR
    mut = QL.fit_levels(Z, y, nonneg=False)                  # МУТАНТ: ограничения сняты
    assert mut[2] < -0.2


def test_all_zero_jump_column_and_bad_input():
    rng = np.random.default_rng(4)
    Z = np.column_stack([np.ones(200), rng.uniform(1, 4, 200), np.zeros(200)])
    b = QL.fit_levels(Z, 1.0 + 0.5 * Z[:, 1])
    assert np.isfinite(b).all() and b[2] == 0.0
    with pytest.raises(ValueError):
        QL.fit_levels(Z[:2], np.ones(2))
    with pytest.raises(ValueError):
        QL.fit_levels(Z, np.r_[0.0, np.ones(199)])           # нулевая RV


def test_positive_and_nonneg_series():
    w = synth_world()
    for wj in (True, False):
        s2, B = QL.run(w, with_j=wj)
        assert (s2 > 0).all() and (B >= 0).all()
        assert B.shape[1] == (5 if wj else 4)


def test_decay_wfit_known_answer_and_weights_matter():
    rng = np.random.default_rng(5)
    n = 400
    Phi = np.column_stack([np.ones(n), rng.normal(size=n)])
    eta = np.where(np.arange(n) < 200, 0.5 + 0.2 * Phi[:, 1], 1.5 + 0.2 * Phi[:, 1])
    y = np.exp(eta)
    flat = QG.wfit(Phi, y, np.ones(n))
    dec = QG.wfit(Phi, y, 0.5 ** ((n - 1 - np.arange(n)) / 20.0))
    assert abs(dec[0] - 1.5) < abs(flat[0] - 1.5) / 5 and abs(dec[1] - 0.2) < 1e-3
    assert np.allclose(QG.wfit(Phi[200:], y[200:], np.ones(200)), [1.5, 0.2], atol=1e-6)


def _series(n=400):
    rv_t = np.ones(n)
    d = np.arange(n).astype("datetime64[D]").astype("datetime64[ns]")
    return rv_t, d, d + np.timedelta64(1, "D")


def test_choose_hl_uses_only_closed_and_closed_plus1_mutant_caught():
    rv_t, d, td = _series()
    rv_t[300] = 1e6                                         # цель строки 300 раскрывается только к d = 301
    S = {h: np.full(400, 1.5) for h in QG.HL}
    S[250.0] = np.full(400, 1.2)                            # на закрытых строках h = 250 лучше, на строке 300 — резко хуже
    a, pick = QG.choose_hl(S, rv_t, d, td)
    assert pick[300] == 0 and a[300] == 1.2                 # на d = 300 строка 300 ещё не закрыта
    _, pick_mut = QG.choose_hl(S, rv_t, d, td, extra=1)     # МУТАНТ «закрыто + 1»
    assert pick_mut[300] != 0
    assert (pick[:QG.MIN_CLOSED] == 3).all()                # < 252 закрытых — база h = ∞


def test_choose_hl_tie_goes_to_larger_h():
    rv_t, d, td = _series()
    S = {h: np.full(400, 1.3) for h in QG.HL}
    _, pick = QG.choose_hl(S, rv_t, d, td)
    assert (pick == 3).all()


def test_scale_c_known_answer_and_closed_plus1_mutant():
    rv_t, d, td = _series()
    rv_t = 2.0 * rv_t
    rv_t[300] = 1e6
    sa, sb = np.ones(400), np.full(400, 1.1)
    out = QG.scale_c(sa, sb, rv_t, d, td)
    assert out[100] == 1.1 and np.isclose(out[300], 2.0) and out[299] == 2.0
    assert not np.isclose(QG.scale_c(sa, sb, rv_t, d, td, extra=1)[300], 2.0)   # МУТАНТ


@pytest.fixture
def small_min_closed(monkeypatch):
    monkeypatch.setattr(QG, "MIN_CLOSED", 10)


RUNS = {"Q1": lambda w: QL.run(w, True), "Q2": lambda w: QL.run(w, False), "Q4": lambda w: QG.run_decay(w), "Q5": lambda w: QG.run_scale(w)}


@pytest.mark.parametrize("name", list(RUNS))
def test_main_truncation_q_rows_unchanged_by_corrupting_future(name, small_min_closed):
    w = synth_world()
    i0 = int(w.el[25])
    assert E.trunc_diff(RUNS[name], w, i0) == 0.0


def test_truncation_catches_future_pair_mutant(small_min_closed, monkeypatch):
    w = synth_world()
    i0 = int(w.el[25])
    orig = W.pairs_idx
    monkeypatch.setattr(W, "pairs_idx", lambda seg, F, i: orig(seg, F, i + 1))   # МУТАНТ: пара (d, d+1) в обучении
    assert E.trunc_diff(RUNS["Q1"], w, i0) > 1e-6
