"""D: LightGBM-пара к XGBoost И1 (`c/fitx.fit_i1x`): та же цель (ln RV − ln RV_m), те же пары, ранняя остановка по последним 252 парам окна,
финальная модель на всех парах с k деревьями, s² = средний квадрат остатка на блоке остановки. HP — те же поля HPX (нового подбора нет)."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np

from models_v2.s4 import fit


def params(hp, seed: int) -> dict:
    d = int(hp.depth)
    return dict(objective="regression", metric="l2", max_depth=d, num_leaves=2 ** d, min_data_in_leaf=max(1, int(hp.min_child)),
                lambda_l2=float(hp.l2), learning_rate=float(hp.lr), bagging_fraction=float(hp.subsample), bagging_freq=1,
                feature_fraction=float(hp.colsample), seed=int(seed), num_threads=1, deterministic=True, force_row_wise=True, verbose=-1)


def _wmean(v, w) -> float:
    return fit._wmean(v, w)


def fit_lgbx(X, y, idx, hp, w=None, mono=None, seed: int = fit.SEED, max_trees: int | None = None) -> dict:
    """Сигнатура как у fit_i1x (mono не используется). -> predict, k, s2_es, s2_tr_prelim, s2_tr, n_train, shrunk, train_idx, es_idx."""
    max_trees = hp.max_trees if max_trees is None else max_trees
    tr, es, shrunk = fit.split(idx)
    g = (lambda a: None if w is None else w[a])
    p = params(hp, seed)
    dtr = lgb.Dataset(X[tr], y[tr], weight=g(tr))
    des = lgb.Dataset(X[es], y[es], weight=g(es), reference=dtr)
    b = lgb.train(p, dtr, max_trees, valid_sets=[des], callbacks=[lgb.early_stopping(fit.PATIENCE, verbose=False)])
    k = max(1, b.best_iteration)
    r_es = y[es] - b.predict(X[es], num_iteration=k)
    r_tr = y[tr] - b.predict(X[tr], num_iteration=k)
    fin = lgb.train(p, lgb.Dataset(X[idx], y[idx], weight=g(idx)), k)
    s2_tr = _wmean((y[idx] - fin.predict(X[idx])) ** 2, g(idx))
    return {"predict": lambda Z: np.asarray(fin.predict(np.atleast_2d(Z)), float), "k": k, "s2_es": _wmean(r_es ** 2, g(es)),
            "s2_tr_prelim": _wmean(r_tr ** 2, g(tr)), "s2_tr": s2_tr, "n_train": len(idx), "shrunk": shrunk, "train_idx": idx, "es_idx": es}
