"""Обучение LightGBM по схеме И1 (`s4/fit`): ранняя остановка по последним 252 парам окна (эмбарго 1, терпение 100), финал на всех парах с k деревьями.

kind: 'ln'    — L2 по ln-цели (ln RV − ln S); predict = f, σ̂² = exp(off + f + s²/2);
      'lvl'   — L1: L2 по уровню y = RV/S; predict = f (уровень), σ̂² = max(S·f, ε);
      'gamma' — L3: objective=gamma (log-link); predict = ln μ, σ̂² = exp(off + ln μ);
      'qlike' — L4: custom QLIKE (F = ln σ̂²), grad/hess из `s5/b1/losses` (проверка конечными разностями — тест); predict = ln μ.
Веса w (L5) — по строкам, как в `lgbfit.fit_lgbx`."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np

from models_v2.s4 import fit
from models_v2.s5.b1 import losses
from models_v2.s6.lgbm.grid import Hp

KINDS = ("ln", "lvl", "gamma", "qlike")
LN_FLOOR = 1e-12


def _wm(v, w) -> float:
    return fit._wmean(v, w)


def _fobj(preds, ds):
    """Custom L4: веса применяются здесь (LightGBM для custom objective их не применяет)."""
    g, h = losses.grad_hess(ds.get_label(), preds)
    wt = ds.get_weight()
    return (g * wt, h * wt) if wt is not None and len(wt) else (g, h)


def _feval(preds, ds):
    wt = ds.get_weight()
    return "qlike", _wm(losses.loss(ds.get_label(), preds), wt if wt is not None and len(wt) else None), False


def _train(kind, p, X, y, idx_tr, idx_es, w, max_trees, f0_tr):
    g = (lambda a: None if w is None else w[a])
    dtr = lgb.Dataset(X[idx_tr], y[idx_tr], weight=g(idx_tr), init_score=None if f0_tr is None else np.full(len(idx_tr), f0_tr))
    des = lgb.Dataset(X[idx_es], y[idx_es], weight=g(idx_es), reference=dtr, init_score=None if f0_tr is None else np.full(len(idx_es), f0_tr))
    kw = {}
    if kind == "qlike":
        p = dict(p, objective=_fobj, metric="None")
        kw["feval"] = _feval
    else:
        p = dict(p, objective="gamma" if kind == "gamma" else "regression", metric="gamma" if kind == "gamma" else "l2")
    b = lgb.train(p, dtr, max_trees, valid_sets=[des], callbacks=[lgb.early_stopping(fit.PATIENCE, verbose=False)], **kw)
    return b, p


def fit_l(X, y, idx, hp: Hp, kind: str = "ln", w=None, seed: int | None = None, max_trees: int | None = None) -> dict:
    """-> predict(Z), k, s2_es (nan вне 'ln'), n_train, k_cap (k == потолок), n_nonfinite_es (нестабильность обучения)."""
    if kind not in KINDS:
        raise ValueError(kind)
    max_trees = hp.max_trees if max_trees is None else max_trees
    tr, es, shrunk = fit.split(idx)
    g = (lambda a: None if w is None else w[a])
    p = hp.params(seed)
    f0 = float(np.log(_wm(y[tr], g(tr)))) if kind == "qlike" else None
    b, p = _train(kind, p, X, y, tr, es, w, max_trees, f0)
    k = max(1, b.best_iteration)
    raw_es = b.predict(X[es], num_iteration=k, raw_score=True)
    s2 = _wm((y[es] - b.predict(X[es], num_iteration=k)) ** 2, g(es)) if kind == "ln" else float("nan")
    f_all = float(np.log(_wm(y[idx], g(idx)))) if kind == "qlike" else 0.0
    dfin = lgb.Dataset(X[idx], y[idx], weight=g(idx), init_score=np.full(len(idx), f_all) if kind == "qlike" else None)
    fin = lgb.train(p, dfin, k)

    def predict(Z):
        Z = np.atleast_2d(Z)
        if kind == "qlike":
            return np.asarray(fin.predict(Z, raw_score=True), float) + f_all
        out = np.asarray(fin.predict(Z), float)
        return np.log(np.maximum(out, LN_FLOOR)) if kind == "gamma" else out
    return {"predict": predict, "k": k, "s2_es": s2, "n_train": len(idx), "shrunk": shrunk, "fin": fin,
            "k_cap": k >= max_trees, "n_nonfinite_es": int((~np.isfinite(raw_es)).sum())}


def sigma2(kind: str, rows, f, s2, off, mult, eps) -> tuple:
    """(σ̂², число срабатываний границы ε) для строк rows; граница — только для 'lvl'."""
    rows, f = np.asarray(rows), np.asarray(f, float)
    if kind == "ln":
        return np.exp(off[rows] + f + np.asarray(s2, float) / 2), 0
    if kind in ("gamma", "qlike"):
        return np.exp(off[rows] + f), 0
    raw = mult[rows] * f
    out = np.maximum(raw, eps[rows])
    return out, int((raw < eps[rows]).sum())
