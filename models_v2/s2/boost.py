"""XGBoost / LightGBM на ln RV_{d+1} (Ж-197 п. 2, Б1–Б6). Модели на диск не сохраняются."""
from __future__ import annotations

import time
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import xgboost as xgb

from models_v2.s2 import features as ft

ES_N, EMBARGO, MIN_TRAIN, MAX_TREES, PATIENCE = 252, 1, 100, 2000, 100
FIXED = dict(lr=0.03, min_leaf=50, subsample=0.8, l2=5.0, seed=42)
SEED = 42


@dataclass(frozen=True)
class Spec:
    lib: str
    loss: str
    har: bool
    mono: bool

    @property
    def name(self) -> str:
        return "%s_%s_%s_%s" % (self.lib, self.loss, "har" if self.har else "nohar", "mono" if self.mono else "free")

    @property
    def cols(self) -> tuple:
        return tuple(range(ft.N_BASE)) + ((ft.I_HAR,) if self.har else ())


def all_specs() -> list:
    return [Spec(lib, loss, h, m) for lib in ("xgb", "lgbm") for loss in ("mse", "hub")
            for h in (False, True) for m in (False, True)]


def pair_index(F: ft.Fp, T: int, spec: Spec) -> np.ndarray:
    """Строки t обучающих пар (x_t, y_{t+1}): t+1 <= T, один сегмент, ядро признаков (и HAR) конечно."""
    t = np.arange(0, T)
    need = list(ft.CORE) + ([ft.I_HAR] if spec.har else [])
    ok = (F.seg[t] == F.seg[t + 1]) & np.isfinite(F.X[t][:, need]).all(axis=1)
    return t[ok]


def split_pairs(t_idx: np.ndarray):
    """(train, es): ES — последние 252 пары, между ними выброшена EMBARGO пара."""
    if len(t_idx) < ES_N + EMBARGO + MIN_TRAIN:
        raise ValueError("мало пар для обучения и ранней остановки: %d" % len(t_idx))
    return t_idx[:-ES_N - EMBARGO], t_idx[-ES_N:]


def _fit_xgb(spec, depth, delta, Xtr, ytr, Xes, yes, mono_vec):
    hub = spec.loss == "hub"
    p = dict(objective="reg:pseudohubererror" if hub else "reg:squarederror", eval_metric="mphe" if hub else "rmse",
             eta=FIXED["lr"], max_depth=depth, min_child_weight=FIXED["min_leaf"], subsample=FIXED["subsample"],
             reg_lambda=FIXED["l2"], tree_method="hist", nthread=1, seed=FIXED["seed"], verbosity=0,
             base_score=float(ytr.mean()))
    if hub:
        p["huber_slope"] = delta
    if any(mono_vec):
        p["monotone_constraints"] = "(" + ",".join(str(v) for v in mono_vec) + ")"
    b = xgb.train(p, xgb.DMatrix(Xtr, label=ytr), MAX_TREES, evals=[(xgb.DMatrix(Xes, label=yes), "es")],
                  early_stopping_rounds=PATIENCE, verbose_eval=False)
    k = b.best_iteration + 1
    g = b.get_score(importance_type="gain")
    imp = np.array([g.get("f%d" % j, 0.0) for j in range(Xtr.shape[1])])
    return (lambda X: b.predict(xgb.DMatrix(X), iteration_range=(0, k))), k, imp


def _fit_lgbm(spec, depth, delta, Xtr, ytr, Xes, yes, mono_vec):
    hub = spec.loss == "hub"
    p = dict(objective="huber" if hub else "regression", metric="huber" if hub else "l2", learning_rate=FIXED["lr"],
             max_depth=depth, num_leaves=2 ** depth, min_data_in_leaf=FIXED["min_leaf"],
             bagging_fraction=FIXED["subsample"], bagging_freq=1, lambda_l2=FIXED["l2"], seed=FIXED["seed"],
             deterministic=True, force_row_wise=True, num_threads=1, verbose=-1)
    if hub:
        p["alpha"] = delta
    if any(mono_vec):
        p["monotone_constraints"] = list(mono_vec)
    ds = lgb.Dataset(Xtr, ytr)
    m = lgb.train(p, ds, MAX_TREES, valid_sets=[lgb.Dataset(Xes, yes, reference=ds)],
                  callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    k = m.best_iteration or MAX_TREES
    return (lambda X: m.predict(X, num_iteration=k)), k, m.feature_importance("gain")


def fit(spec: Spec, depth: int, delta: float, Xtr, ytr, Xes, yes):
    """-> (predict, n_trees, gain, s2_es, s2_tr). s2: средний квадрат остатка ln (ES-блок / обучение)."""
    mono_vec = [1 if (spec.mono and j in ft.MONO) else 0 for j in range(Xtr.shape[1])]
    f = _fit_xgb if spec.lib == "xgb" else _fit_lgbm
    pred, k, imp = f(spec, depth, delta, Xtr, ytr, Xes, yes, mono_vec)
    s2_es = float(np.mean((yes - pred(Xes)) ** 2))
    s2_tr = float(np.mean((ytr - pred(Xtr)) ** 2))
    return pred, k, imp, s2_es, s2_tr


def run_block(F: ft.Fp, spec: Spec, depth: int, delta: float, T: int, d_idx, variant: str = "work") -> dict:
    """Модель по парам t+1 <= T; прогноз ln RV_{d+1} для строк d_idx. variant: work|oracle|perm|full."""
    t0 = time.time()
    t_idx = pair_index(F, T, spec)
    tr, es = split_pairs(t_idx)
    cols = list(spec.cols)
    Xtr, ytr, Xes, yes = F.X[tr][:, cols], F.lnrv[tr + 1], F.X[es][:, cols], F.lnrv[es + 1]
    Xnew = F.X[np.asarray(d_idx)][:, cols]
    if variant == "oracle":  # (i) цель доступна как признак
        Xtr, Xes = np.column_stack([Xtr, ytr]), np.column_stack([Xes, yes])
        Xnew = np.column_stack([Xnew, F.lnrv[np.asarray(d_idx) + 1]])
    elif variant == "perm":  # (iii) строки X перемешаны внутри окна
        perm = np.random.default_rng(SEED + int(T)).permutation(len(tr) + len(es))
        Xall = np.vstack([Xtr, Xes])[perm]
        Xtr, Xes = Xall[:len(tr)], Xall[len(tr):]
    pred, k, imp, s2_es, s2_tr = fit(spec, depth, delta, Xtr, ytr, Xes, yes)
    imp = imp[:len(cols)]
    return {"f": pred(Xnew), "k": k, "imp": imp / imp.sum() if imp.sum() > 0 else imp, "no_split": bool(imp.sum() == 0),
            "s2_es": s2_es,
            "s2_tr": s2_tr, "n_train": len(tr), "sec": time.time() - t0}


def retrain_index(dates: np.ndarray, i: int) -> int:
    """Строка первого годного дня месяца строки i (день переобучения; данные <= этого дня)."""
    m = dates.astype("datetime64[M]")
    return int(np.flatnonzero(m == m[i])[0])


def blocks(F: ft.Fp, grid, daily: bool = True) -> dict:
    """{T: [d, …]} — к какому переобучению относится каждая дата сетки (ежедневно: T = d)."""
    out = {}
    for d in grid:
        out.setdefault(int(d) if daily else retrain_index(F.dates, int(d)), []).append(int(d))
    return out


def forecast_at(F: ft.Fp, spec: Spec, depth: int, delta: float, d: int, daily: bool = True) -> float:
    """ln f для строки d: модель, обученная в день d (daily) или в первый годный день месяца d."""
    T = d if daily else retrain_index(F.dates, d)
    return float(run_block(F, spec, depth, delta, T, [d])["f"][0])
