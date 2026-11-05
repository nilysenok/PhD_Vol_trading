"""G2 D: LGBM-пара и окна/веса — детерминизм, seed, ранняя остановка по блоку, окно и полураспад (MD10). Синтетика, без сети."""
import numpy as np
import pytest

from models_v2.s4 import fit
from models_v2.s5.c import fitx
from models_v2.s5.d import engine_d, lgbfit

HP = fitx.HPX(2, 20.0, 5.0, 0.1, 0.7, 0.6, 200, 0.0, 0.0)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(900, 6))
    y = 0.8 * X[:, 0] - 0.4 * X[:, 1] ** 2 + rng.normal(0, 0.5, 900)
    return X, y


def test_lgbm_deterministic_bitwise(data):
    X, y = data
    idx = np.arange(700)
    a, b = (lgbfit.fit_lgbx(X, y, idx, HP, seed=42) for _ in range(2))
    assert a["predict"](X[700:]).tobytes() == b["predict"](X[700:]).tobytes() and a["k"] == b["k"] and a["s2_es"] == b["s2_es"]


def test_lgbm_seed_acts_when_subsampling(data):
    X, y = data
    idx = np.arange(700)
    p = [lgbfit.fit_lgbx(X, y, idx, HP, seed=s)["predict"](X[700:]) for s in (42, 43)]
    assert not np.array_equal(p[0], p[1])


def test_lgbm_ignores_rows_outside_window_and_uses_es_block_only(data):
    X, y = data
    idx = np.arange(700)
    y2 = y.copy()
    y2[700:] += 50.0
    X2 = X.copy()
    X2[700:, 0] *= 100.0   # правка строк после окна: прогноз на фиксированных Z не меняется
    a, b = (lgbfit.fit_lgbx(Xx, yy, idx, HP)["predict"](X[:5]) for Xx, yy in ((X, y), (X2, y2)))
    assert a.tobytes() == b.tobytes()
    m = lgbfit.fit_lgbx(X, y, idx, HP)
    tr, es, _ = fit.split(idx)
    assert np.array_equal(m["es_idx"], es) and len(es) == fit.ES_N and es.min() - tr.max() > fit.EMBARGO and np.isfinite(m["s2_es"]) and m["s2_es"] > 0 and m["k"] >= 1


def test_lgbm_learns_signal(data):
    X, y = data
    m = lgbfit.fit_lgbx(X, y, np.arange(700), HP)
    r = np.corrcoef(m["predict"](X[700:]), y[700:])[0, 1]
    assert r > 0.5 and m["s2_es"] < y.var()


def test_params_map_hp_axes():
    p = lgbfit.params(fitx.HPX(3, 41.7, 11.0, 0.024, 0.9, 0.5, 300), 44)
    assert (p["max_depth"], p["num_leaves"], p["min_data_in_leaf"], p["lambda_l2"], p["learning_rate"], p["bagging_fraction"], p["feature_fraction"], p["seed"]) == \
        (3, 8, 41, 11.0, 0.024, 0.9, 0.5, 44) and p["num_threads"] == 1 and p["deterministic"]


def test_md10_window_and_halflife(monkeypatch, data):
    assert (engine_d.W1008, engine_d.HL504) == (1008, 504)
    X, y = data

    class F:
        rv = np.zeros(len(y))
        seg = np.zeros(len(y) + 1, int)

    class Dz:
        pass
    dz = Dz()
    dz.X, dz.y = X, y
    got = {}

    def spy(X_, y_, idx, hp, w, mono, seed):
        got.update(idx=idx, w=w)
        return {"predict": None}
    monkeypatch.setattr(engine_d.runner, "pairs_upto", lambda dz_, F_, T: np.arange(T))
    monkeypatch.setattr(engine_d.fitx, "fit_i1x", spy)
    engine_d.day_fit(F, dz, 800, HP, 42, "xgb", win=300)
    assert len(got["idx"]) == 300 and got["idx"].min() == 500 and got["w"] is None
    engine_d.day_fit(F, dz, 800, HP, 42, "xgb", hl=engine_d.HL504)
    assert len(got["idx"]) == 800 and np.isclose(got["w"][800 - 504], 0.5) and np.isclose(got["w"][800], 1.0) and got["w"][0] < got["w"][799]
    engine_d.day_fit(F, dz, 800, HP, 42, "xgb")
    assert len(got["idx"]) == 800 and got["w"] is None
