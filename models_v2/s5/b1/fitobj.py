"""Обучение с разными потерями по схеме И1 (`s4/fit.fit_i1`): ES — последние 252 пары, эмбарго 1, финал на всех парах окна с n* деревьями.

loss: 'l2' / 'mse' — fit_i1 как есть (MSE ln-цели / MSE уровня); 'gamma' — XGBoost reg:gamma (log-link);
'qlike' — пользовательская потеря L4; 'lgb_gamma' — LightGBM gamma. predict(Z): l2 — f (ln-отклонение), mse — уровень,
gamma / qlike / lgb_gamma — f = ln μ (μ — прогноз на шкале y).
Возврат как у fit_i1 плюс tr_loss / es_loss — средняя потеря конфигурации на предварительной модели (обучение / блок остановки).
"""
from __future__ import annotations

import numpy as np
import xgboost as xgb

from models_v2.s4 import fit
from models_v2.s5.b1 import losses

NATIVE = {"gamma", "qlike", "lgb_gamma"}


def _wm(v, w):
    return float(np.average(v, weights=w)) if w is not None else float(np.mean(v))


def fit_loss(X, y, idx, hp: fit.HP, loss: str, w=None) -> dict:
    if loss in ("l2", "mse"):
        m = fit.fit_i1(X, y, idx, hp, w)
        return {**m, "tr_loss": m["s2_tr_prelim"], "es_loss": m["s2_es"]}
    if loss == "lgb_gamma":
        return _lgb(X, y, idx, hp, w)
    return _xgb(X, y, idx, hp, loss, w)


def _xgb(X, y, idx, hp, loss, w) -> dict:
    tr, es, shrunk = fit.split(idx)
    g = (lambda a: None if w is None else w[a])
    gamma = loss == "gamma"

    def dm(a, f0=None):
        d = xgb.DMatrix(X[a], label=y[a], weight=g(a))
        if f0 is not None:
            d.set_base_margin(np.full(len(a), f0))
        return d

    def params(a):
        p = fit._params(hp, None, _wm(y[a], g(a)))
        if gamma:
            p.update(objective="reg:gamma", eval_metric="gamma-nloglik")
        else:
            p.pop("eval_metric")
        return p

    kw = {} if gamma else {"obj": losses.qlike_obj, "custom_metric": losses.qlike_metric}
    f_tr = None if gamma else float(np.log(_wm(y[tr], g(tr))))
    dtr, des = dm(tr, f_tr), dm(es, f_tr)
    b = xgb.train(params(tr), dtr, hp.max_trees, evals=[(des, "es")], early_stopping_rounds=fit.PATIENCE, verbose_eval=False, **kw)
    k = b.best_iteration + 1
    to_f = (lambda a, d: np.log(b.predict(d, iteration_range=(0, k)))) if gamma else (lambda a, d: b.predict(d, iteration_range=(0, k)))
    tr_l = _wm(losses.loss(y[tr], to_f(tr, dtr)), g(tr))
    es_l = _wm(losses.loss(y[es], to_f(es, des)), g(es))
    f_all = None if gamma else float(np.log(_wm(y[idx], g(idx))))
    fin = xgb.train(params(idx), dm(idx, f_all), k, **kw)

    def predict(Z):
        Z = np.atleast_2d(Z)
        d = xgb.DMatrix(Z) if gamma else xgb.DMatrix(Z, base_margin=np.full(len(Z), f_all))
        p = fin.predict(d).astype(float)
        return np.log(p) if gamma else p
    return {"predict": predict, "k": k, "s2_es": float("nan"), "s2_tr_prelim": float("nan"), "n_train": len(idx), "shrunk": shrunk,
            "train_idx": idx, "es_idx": es, "tr_loss": tr_l, "es_loss": es_l}


def _lgb(X, y, idx, hp, w) -> dict:
    import lightgbm as lgb
    tr, es, shrunk = fit.split(idx)
    g = (lambda a: None if w is None else w[a])
    p = dict(objective="gamma", metric="gamma", max_depth=hp.depth, num_leaves=2 ** hp.depth, min_data_in_leaf=int(hp.min_child),
             lambda_l2=hp.l2, learning_rate=hp.lr, bagging_fraction=hp.subsample, bagging_freq=1, feature_fraction=hp.colsample,
             seed=fit.SEED, num_threads=1, deterministic=True, force_row_wise=True, verbose=-1)
    dtr = lgb.Dataset(X[tr], y[tr], weight=g(tr))
    des = lgb.Dataset(X[es], y[es], weight=g(es), reference=dtr)
    b = lgb.train(p, dtr, hp.max_trees, valid_sets=[des], callbacks=[lgb.early_stopping(fit.PATIENCE, verbose=False)])
    k = max(1, b.best_iteration)
    f_tr = np.log(b.predict(X[tr], num_iteration=k))
    f_es = np.log(b.predict(X[es], num_iteration=k))
    fin = lgb.train(p, lgb.Dataset(X[idx], y[idx], weight=g(idx)), k)
    return {"predict": lambda Z: np.log(fin.predict(np.atleast_2d(Z))).astype(float), "k": k, "s2_es": float("nan"),
            "s2_tr_prelim": float("nan"), "n_train": len(idx), "shrunk": shrunk, "train_idx": idx, "es_idx": es,
            "tr_loss": _wm(losses.loss(y[tr], f_tr), g(tr)), "es_loss": _wm(losses.loss(y[es], f_es), g(es))}
