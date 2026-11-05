"""B1-3 (Ж-204): L4 (прямой QLIKE) — градиент/гессиан численно, минимум на синтетике, тождество с metrics.qlike, XGBoost custom/gamma, LightGBM gamma."""
import numpy as np
import pytest
import xgboost as xgb
from scipy.optimize import minimize_scalar

from models_v2 import metrics
from models_v2.s4 import fit
from models_v2.s5.b1 import fitobj, losses

RNG = np.random.default_rng(5)
HP = fit.HP(3, 10.0, 2.0, 0.1, 0.8, 1.0, 120)


def sample(n=400):
    return RNG.gamma(3.0, 1 / 3.0, n) * np.exp(RNG.normal(0, 1, n)), RNG.normal(0, 1, n)


def rel_ok(a, b, tol=1e-6):
    return np.max(np.abs(a - b) / (1 + np.abs(b))) <= tol


def test_grad_hess_match_finite_differences():
    y, F = sample()
    g, h = losses.grad_hess(y, F)
    gn, hn = losses.numeric_grad_hess(y, F, 1e-4)
    assert rel_ok(g, gn) and rel_ok(h, hn)


def test_mutation_sign_of_gradient_is_caught():
    y, F = sample()
    g, h = losses.grad_hess(y, F)
    gn, hn = losses.numeric_grad_hess(y, F, 1e-4)
    assert not rel_ok(-g, gn)                       # M-B1-1: знак градиента
    assert not rel_ok(np.exp(-F), hn)               # M-B1-2: hess без y


def test_constant_minimum_is_log_mean_y():
    y, _ = sample(2000)
    r = minimize_scalar(lambda F: losses.loss(y, np.full(len(y), F)).mean(), bracket=(-2, 2), tol=1e-12)
    assert abs(r.x - np.log(y.mean())) < 1e-3


def test_loss_minus_qlike_does_not_depend_on_forecast():
    y, _ = sample()
    d = [losses.loss(y, np.log(f)) - metrics.qlike(y, f) for f in (0.7 * np.ones(len(y)), 2.3 * np.ones(len(y)))]
    assert np.max(np.abs(d[0] - d[1])) < 1e-12
    assert np.max(np.abs(d[0] - (1 + np.log(y)))) < 1e-12   # ровно 1 + ln y


def test_qlike_scale_invariant_so_ratio_target_equals_level_target():
    y, _ = sample()
    f, S = np.exp(RNG.normal(0, 1, len(y))), np.exp(RNG.normal(0, 2, len(y)))
    assert np.max(np.abs(metrics.qlike(y * S, f * S) - metrics.qlike(y, f))) < 1e-12


def test_weights_applied_by_custom_objective():
    y, F = sample(50)
    dm = xgb.DMatrix(np.zeros((50, 1)), label=y, weight=np.full(50, 3.0))
    g, h = losses.qlike_obj(F, dm)
    g0, h0 = losses.grad_hess(y, F)
    assert np.allclose(g, 3 * g0) and np.allclose(h, 3 * h0)


def data(n=700, shape_noise=0.3):
    x = RNG.uniform(-1, 1, (n, 3))
    mu = np.exp(0.8 * x[:, 0] - 0.5 * x[:, 1])
    return x, mu * RNG.gamma(1 / shape_noise ** 2, shape_noise ** 2, n), mu


@pytest.mark.parametrize("loss", ["qlike", "gamma", "lgb_gamma"])
def test_native_losses_recover_known_dependence(loss):
    x, y, mu = data()
    idx = np.arange(len(y))
    m = fitobj.fit_loss(x, y, idx, HP, loss)
    f = m["predict"](x)
    assert np.corrcoef(f, np.log(mu))[0, 1] >= 0.9
    assert np.isfinite(m["tr_loss"]) and np.isfinite(m["es_loss"])


def test_qlike_and_gamma_are_same_loss_on_ratio_target():
    x, y, _ = data()
    idx = np.arange(len(y))
    a = fitobj.fit_loss(x, y, idx, HP, "qlike")["predict"](x)
    b = fitobj.fit_loss(x, y, idx, HP, "gamma")["predict"](x)
    assert np.corrcoef(a, b)[0, 1] > 0.99


def test_custom_metric_sees_margin_with_base_margin():
    x, y, _ = data(300)
    y = 5.0 * y
    f0 = float(np.log(y.mean()))
    dm = xgb.DMatrix(x, label=y)
    dm.set_base_margin(np.full(300, f0))
    er = {}
    p = fit._params(HP, None, 0.5)
    p.pop("eval_metric")
    b = xgb.train(p, dm, 20, evals=[(dm, "es")], obj=losses.qlike_obj, custom_metric=losses.qlike_metric, evals_result=er, verbose_eval=False)
    manual = float(np.mean(losses.loss(y, b.predict(dm))))
    assert abs(er["es"]["qlike"][-1] - manual) < 1e-5   # margin XGBoost — float32 (расхождение ~4e-7); без base_margin разность была бы порядка |f0|
    assert abs(f0) > 1.0


def test_gamma_nloglik_equals_y_over_mu_plus_ln_mu():
    x, y, _ = data(300)
    er = {}
    p = fit._params(HP, None, float(y.mean()))
    p.update(objective="reg:gamma", eval_metric="gamma-nloglik")
    dm = xgb.DMatrix(x, label=y)
    b = xgb.train(p, dm, 20, evals=[(dm, "es")], evals_result=er, verbose_eval=False)
    mu = b.predict(dm).astype(float)
    assert abs(er["es"]["gamma-nloglik"][-1] - np.mean(y / mu + np.log(mu))) < 1e-5
