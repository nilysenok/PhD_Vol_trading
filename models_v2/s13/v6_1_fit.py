"""V6-1a, G12: обучатели поправки g (план V6, п. 2). F = H·exp(clip(g, −1, 1)), остаток e = ln RV_{t⁺} − ln H. QLIKE(RV, F) = exp(e − g) − (e − g) − 1: выпукла по g; градиент 1 − exp(e − g), гессиан exp(e − g).
LIN: g = θ₀ + θ'z, минимум mean QLIKE + λ‖θ‖² (Ньютон с делением шага; θ₀ не штрафуется — толкование плана «λ‖θ‖²» на наклонах). λ — вложенный PIT-выбор: сетка {1e-3, 1e-2, 1e-1, 1}; обучение — все закрытые пары, кроме последних 252,
проверка — последние 252 (QLIKE с clip |g| ≤ 1); затем переобучение на ВСЕХ закрытых парах. Нормировка z — среднее и СКО по тем же парам, на которых идёт обучение.
GBM: LightGBM с собственной целью QLIKE по g (g₀ = 0): num_leaves 4, learning_rate 0,03, min_data_in_leaf 50, feature_fraction 0,8, без бэггинга, seed 0, ≤ 300 деревьев. Число деревьев — лучшая итерация на последних 252 закрытых парах
(обучение на остальных; ранняя остановка без терпения), затем переобучение на ВСЕХ закрытых парах с этим числом деревьев (мутации № 99, 100)."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np

GRID = (1e-3, 1e-2, 1e-1, 1.0)
CLIP, HOLD, MAX_TREES, MIN_TRAIN = 1.0, 252, 300, 252 + 60
GBM_PARAMS = dict(num_leaves=4, learning_rate=0.03, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=1.0, bagging_freq=0, seed=0, verbose=-1, num_threads=1, deterministic=True, force_row_wise=True)


def qlike_g(e, g) -> np.ndarray:
    u = np.asarray(e, float) - np.asarray(g, float)
    return np.exp(u) - u - 1.0


def grad_hess(e, g) -> tuple:
    w = np.exp(np.asarray(e, float) - np.asarray(g, float))
    return 1.0 - w, w


def clip_g(g) -> np.ndarray:
    return np.clip(np.asarray(g, float), -CLIP, CLIP)


def forecast(h, g) -> np.ndarray:
    """F = H·exp(clip(g, −1, 1)); при g ≡ 0 F = H побитно (exp(0) = 1)."""
    return np.asarray(h, float) * np.exp(clip_g(g))


def norm_stats(X: np.ndarray) -> tuple:
    """Среднее и СКО (ddof = 0) по строкам X — закрытым парам обучения. МУТАНТ «по всей выборке» подменяет эту функцию в тесте."""
    sd = X.std(axis=0)
    return X.mean(axis=0), np.where(sd > 1e-12, sd, 1.0)


def _design(Z: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(Z)), Z])


def _obj(A, e, th, lam) -> float:
    return float(qlike_g(e, np.einsum("ij,j->i", A, th)).mean() + lam * (th[1:] ** 2).sum())


def fit_lin_fixed(Z: np.ndarray, e: np.ndarray, lam: float, iters: int = 60) -> np.ndarray:
    """θ (k+1) для фиксированного λ: Ньютон на mean QLIKE + λ‖θ‖², шаг делится пополам до убывания цели."""
    n, k = Z.shape
    A = _design(Z)
    pen = np.r_[0.0, np.full(k, lam)]
    th = np.zeros(k + 1)
    for _ in range(iters):
        g1, h1 = grad_hess(e, np.einsum("ij,j->i", A, th))
        gr = np.einsum("ij,i->j", A, g1) / n + 2 * pen * th
        hs = np.einsum("ij,i,ik->jk", A, h1, A) / n + 2 * np.diag(pen)
        if np.abs(gr).max() < 1e-11:
            break
        step, t, f0 = np.linalg.solve(hs, gr), 1.0, _obj(A, e, th, lam)
        while _obj(A, e, th - t * step, lam) > f0 and t > 1e-12:
            t /= 2
        th = th - t * step
    return th


def split_es(n: int, hold: int = HOLD) -> tuple:
    """Индексы (обучение, проверка/ранняя остановка): проверка — последние `hold` закрытых пар. МУТАНТ № 99 (блок остановки в обучении) подменяет эту функцию."""
    return np.arange(n - hold), np.arange(n - hold, n)


def final_idx(n: int, hold: int = HOLD) -> np.ndarray:
    """Строки итогового переобучения: ВСЕ закрытые пары. МУТАНТ № 100 (без последних 252) подменяет эту функцию."""
    return np.arange(n)


class LinModel:
    def __init__(self, mu, sd, theta, lam, scores):
        self.mu, self.sd, self.theta, self.lam, self.scores = mu, sd, theta, lam, scores

    def g(self, X: np.ndarray) -> np.ndarray:
        return np.einsum("ij,j->i", _design((np.asarray(X, float) - self.mu) / self.sd), self.theta)


def select_lambda(X: np.ndarray, e: np.ndarray, grid=GRID, hold: int = HOLD) -> tuple:
    """Индекс λ по QLIKE(clip g) на последних `hold` парах; обучение (и его нормировка) — только на предыдущих. -> (idx, scores)."""
    tr, va = split_es(len(e), hold)
    mu, sd = norm_stats(X[tr])
    Zt, Zv = (X[tr] - mu) / sd, (X[va] - mu) / sd
    sc = [float(qlike_g(e[va], clip_g(np.einsum("ij,j->i", _design(Zv), fit_lin_fixed(Zt, e[tr], lam)))).mean()) for lam in grid]
    return int(np.argmin(sc)), sc


def fit_lin(X: np.ndarray, e: np.ndarray, lam_shift: int = 0) -> LinModel:
    """LIN на закрытых парах (X, e). lam_shift = ±1 — соседняя точка сетки (чувствительность, не конфигурация)."""
    if len(e) < MIN_TRAIN:
        raise ValueError("мало закрытых пар: %d < %d" % (len(e), MIN_TRAIN))
    i, sc = select_lambda(X, e)
    lam = GRID[int(np.clip(i + lam_shift, 0, len(GRID) - 1))]
    mu, sd = norm_stats(X)
    return LinModel(mu, sd, fit_lin_fixed((X - mu) / sd, e, lam), lam, sc)


def _train(X, e, n_trees: int):
    def fobj(preds, data):
        return grad_hess(data.get_label(), preds)
    return lgb.train({**GBM_PARAMS, "objective": fobj}, lgb.Dataset(X, label=e, free_raw_data=False), num_boost_round=n_trees)


class GbmModel:
    def __init__(self, booster, n_trees, info):
        self.booster, self.n_trees, self.info = booster, n_trees, info

    def g(self, X: np.ndarray) -> np.ndarray:
        return self.booster.predict(np.asarray(X, float), raw_score=True)


def fit_gbm(X: np.ndarray, e: np.ndarray, trees_mult: float = 1.0) -> GbmModel:
    """Ранняя остановка на последних 252 закрытых парах (обучение — на остальных), затем переобучение на всех с этим числом деревьев. trees_mult — ×0,5/×2 для чувствительности."""
    n = len(e)
    if n < MIN_TRAIN:
        raise ValueError("мало закрытых пар: %d < %d" % (n, MIN_TRAIN))
    tr, es = split_es(n)
    b = _train(X[tr], e[tr], MAX_TREES)
    sc = [float(qlike_g(e[es], clip_g(b.predict(X[es], num_iteration=k, raw_score=True))).mean()) for k in range(1, MAX_TREES + 1)]
    best = int(np.argmin(sc)) + 1
    k = max(1, int(round(best * trees_mult)))
    fi = final_idx(n)
    info = dict(n_train=len(tr), n_es=len(es), n_final=len(fi), train_max=int(tr.max()), es_min=int(es.min()), final_max=int(fi.max()), best=best, n_trees=k)
    return GbmModel(_train(X[fi], e[fi], k), k, info)
