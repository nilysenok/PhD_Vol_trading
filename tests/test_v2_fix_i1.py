"""И1 (Ж-200): финальная модель обучена до d включительно; n* — на блоке остановки; мутации M1–M3 ловятся."""
import numpy as np
import pytest

from models_v2.s4 import fit, runner

HP = fit.HP(3, 10.0, 1.0, 0.1, 1.0)


def synth(n=900, seed=0, noise=0.1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + noise * rng.normal(size=n)   # y[t] — цель пары (x_t, y_t)
    return X, y, np.zeros(n, int)


class Dz:
    def __init__(self, X, y):
        self.X, self.y, self.off = X, y, np.zeros(len(y))


class Fx:
    def __init__(self, n):
        self.seg, self.dates, self.rv = np.zeros(n, int), np.arange(n).astype("datetime64[D]"), np.ones(n)


def check_train(idx_final, T):
    """Контракт И1: последняя доступная пара (t+1 = T) в обучении, пар с t+1 > T нет."""
    assert idx_final.max() + 1 == T, "нет последней пары t+1 = d"
    assert (idx_final + 1 <= T).all(), "в обучении пара из будущего"


def test_final_model_trained_through_d_and_known_answer():
    X, y, seg = synth()
    T = 800
    idx = fit.pair_index(seg, X, y, (0, 1, 2, 3), T)
    m = fit.fit_i1(X, y, idx, HP)
    check_train(m["train_idx"], T)
    assert set(m["es_idx"]) <= set(m["train_idx"])        # блок остановки вошёл в финальную модель
    assert m["n_train"] == len(idx) == T                  # все пары окна, без выброшенных 252
    te = np.arange(T, 899)
    err = np.sqrt(np.mean((m["predict"](X[te]) - (y[te])) ** 2))
    assert err < 0.25                                     # шум 0,1: модель учит связь


def test_es_block_is_last_252_with_embargo_and_nstar_capped():
    X, y, seg = synth(600)
    idx = fit.pair_index(seg, X, y, (0, 1, 2, 3), 590)
    tr, es, shrunk = fit.split(idx)
    assert not shrunk and len(es) == 252 and es[-1] == idx[-1]
    assert idx[len(tr)] not in set(tr) | set(es)          # эмбарго: одна пара между блоками
    assert 1 <= fit.fit_i1(X, y, idx, HP)["k"] <= fit.MAX_TREES


def test_s2_from_es_block_outside_prelim_training_pure_noise():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(700, 4)), rng.normal(size=700)
    idx = fit.pair_index(np.zeros(700, int), X, y, (0, 1, 2, 3), 690)
    m = fit.fit_i1(X, y, idx, fit.HP(5, 2.0, 0.0, 0.1, 1.0))
    assert m["s2_es"] > m["s2_tr_prelim"]                 # s² вне обучения предварительной модели (M3 дал бы меньше)
    assert m["s2_es"] == pytest.approx(1.0, abs=0.35)


def test_mutation_m1_without_last_252_caught():
    X, y, seg = synth()
    idx = fit.pair_index(seg, X, y, (0, 1, 2, 3), 800)
    bad = fit.fit_i1(X, y, idx[:-252], HP)                # M1
    with pytest.raises(AssertionError):
        check_train(bad["train_idx"], 800)


def test_mutation_m2_pair_from_future_caught():
    X, y, seg = synth()
    idx = fit.pair_index(seg, X, y, (0, 1, 2, 3), 801)    # M2: окно до d+1
    with pytest.raises(AssertionError):
        check_train(fit.fit_i1(X, y, idx, HP)["train_idx"], 800)


def test_block_fit_contract_through_runner():
    X, y, _ = synth()
    r = runner.block_fit(Fx(900), Dz(X, y), HP, 800, [800], False, None, "work")
    assert r["last_t"] == 799 and r["n_train"] == 800


def test_weights_halflife_and_monotone():
    w = fit.weights(np.arange(2000), 1999)
    assert w[1999] == 1.0 and w[1999 - 504] == pytest.approx(0.5) and np.all(np.diff(w) > 0)   # M7: убывание вспять ловится
    X, y, seg = synth()
    idx = fit.pair_index(seg, X, y, (0, 1, 2, 3), 800)
    a, b = fit.fit_i1(X, y, idx, HP), fit.fit_i1(X, y, idx, HP, np.ones(len(y)))
    assert np.array_equal(a["predict"](X[800:]), b["predict"](X[800:]))   # единичные веса = без весов


def test_deterministic_two_fits_bitwise():
    X, y, seg = synth()
    idx = fit.pair_index(seg, X, y, (0, 1, 2, 3), 800)
    a, b = fit.fit_i1(X, y, idx, HP), fit.fit_i1(X, y, idx, HP)
    assert a["predict"](X[800:]).tobytes() == b["predict"](X[800:]).tobytes()
