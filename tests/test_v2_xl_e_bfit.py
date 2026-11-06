"""Волна E, тип B (GATES_E G2): QLIKE-объектив с offset ln HAR — градиент/гессиан = конечным разностям и встроенному gamma (XGBoost, LightGBM);
положительность σ̂² по построению; будущие цели не меняют прогноз на d; мутация знака гессиана ловится."""
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pytest
import xgboost as xgb

from models_v2.s6.e import bfit
from models_v2.s6.lgbm.grid import Hp
from models_v2.s6.xgb import c_space


def data(n=700, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    lb = rng.normal(0.0, 0.3, n)
    g = 0.5 * np.tanh(X[:, 0])                          # известная поправка
    y = np.exp(lb + g) * rng.gamma(20.0, 1 / 20.0, n)   # E[y] = HAR·e^g, относительный шум ≈ 22 %
    return X, lb, y, g


def test_grad_hess_equal_finite_differences():
    rng = np.random.default_rng(0)
    y, m = np.exp(rng.normal(0, 1, 50)), rng.normal(0, 1, 50)
    loss = lambda mm: mm + y * np.exp(-mm)  # noqa: E731
    h = 1e-5
    g, hs = bfit.qlike_gh(y, m)
    assert np.allclose(g, (loss(m + h) - loss(m - h)) / (2 * h), atol=1e-7)
    assert np.allclose(hs, (loss(m + h) - 2 * loss(m) + loss(m - h)) / h ** 2, atol=1e-4)
    assert (hs > 0).all()
    gm, _ = bfit.qlike_gh(y, m)
    assert not np.allclose(gm, -g)                                # мутация «перепутан знак» отличима


def test_xgb_gamma_base_margin_equals_custom_qlike_objective():
    X, lb, y, _ = data()
    P = dict(max_depth=2, eta=0.1, nthread=1, seed=42, base_score=1.0, tree_method="hist", verbosity=0, min_child_weight=1, reg_lambda=1.0)
    a = xgb.train(dict(P, objective="reg:gamma"), xgb.DMatrix(X, label=y, base_margin=lb), 20)
    d = xgb.DMatrix(X, label=y, base_margin=lb)
    b = xgb.train(dict(P, objective="reg:squarederror"), d, 20, obj=lambda p, dt: bfit.qlike_gh(dt.get_label(), p))
    assert np.allclose(a.predict(d, output_margin=True), b.predict(d, output_margin=True), atol=1e-4)
    bad = xgb.train(dict(P, objective="reg:squarederror"), d, 20, obj=lambda p, dt: tuple(-v for v in bfit.qlike_gh(dt.get_label(), p)))
    assert not np.allclose(a.predict(d, output_margin=True), bad.predict(d, output_margin=True), atol=1e-2)      # мутация знака ловится


def test_lgbm_gamma_init_score_equals_custom_qlike_objective():
    X, lb, y, _ = data()
    P = dict(num_leaves=4, learning_rate=0.1, min_data_in_leaf=20, num_threads=1, deterministic=True, force_row_wise=True, verbose=-1, seed=1)
    a = lgb.train(dict(P, objective="gamma"), lgb.Dataset(X, y, init_score=lb), 20)
    b = lgb.train(dict(P, objective=lambda p, ds: bfit.qlike_gh(ds.get_label(), p + lb)), lgb.Dataset(X, y, init_score=np.zeros(len(y))), 20)
    assert np.allclose(a.predict(X, raw_score=True), b.predict(X, raw_score=True), atol=1e-6)


def test_fit_recovers_known_correction_and_sigma_positive():
    X, lb, y, g = data()
    idx = np.arange(600)
    hx = c_space.HPC(2, 5.0, 1.0, 0.1, 1.0, 1.0, 300)
    hl = Hp(4, 20, 1.0, 0.1, 1.0, 1.0, -1, 300)
    for fn, hp in ((bfit.fit_xq, hx), (bfit.fit_lq, hl)):
        m = fn(X, lb, y, idx, hp, 42)
        pred = m["predict"](X[600:], lb[600:])               # ln σ̂²
        assert np.isfinite(pred).all() and (np.exp(pred) > 0).all()
        assert np.corrcoef(pred - lb[600:], g[600:])[0, 1] > 0.8     # g найдена, а не HAR воспроизведён
        assert abs(np.mean(pred - lb[600:] - g[600:])) < 0.1


def test_future_targets_do_not_change_forecast_on_d():
    X, lb, y, _ = data(n=520)
    F = SimpleNamespace(seg=np.zeros(520, int), dates=np.datetime64("2016-01-01") + np.arange(520).astype("timedelta64[D]"))
    hx = c_space.HPC(2, 5.0, 1.0, 0.1, 1.0, 1.0, 100)
    d = 450
    hpy = {Y: hx for Y in range(2016, 2019)}
    bfit.init(F, X, lb, y, np.log(y), y, hpy, "xq", 42)
    a = bfit.day_job(d)[1]["lnmu"]
    y2 = y.copy()
    y2[d:] *= 50.0                                            # цели строк t >= d (RV_{t+1}) недоступны на d
    bfit.init(F, X, lb, y2, np.log(y2), y2, hpy, "xq", 42)
    assert bfit.day_job(d)[1]["lnmu"] == a
    y3 = y.copy()
    y3[d - 5] *= 50.0                                         # закрытая цель обязана менять прогноз (тест чувствителен)
    bfit.init(F, X, lb, y3, np.log(y3), y3, hpy, "xq", 42)
    assert bfit.day_job(d)[1]["lnmu"] != a
