"""V4-CLD: регрессии условного ожидания D = QLIKE(H) − QLIKE(M) на признаки z (стандартизованы по окну оценки). Гиперпараметры фиксированы объявлением, не выбираются:
CLD1 гребень (λ = 250 пар-эквивалентов: штраф λ‖β‖² при единичной дисперсии столбцов, свободный член не штрафуется), отображение w = σ(D̂/ŝ), ŝ — с. о. остатка на окне оценки;
CLD2 та же регрессия, w = clip(0,5 + D̂/(2c), 0, 1), c — 90-й процентиль |D| окна оценки; CLD3 Хьюбер (δ = 1,345·ŝ, ŝ — с. о. остатка гребневой подгонки того же окна; IRLS) + тот же гребень, отображение как CLD1;
CLD4 аддитивная: на признак — кубический B-сплайн, 4 степени свободы (1 внутренний узел в медиане окна, границы — min/max окна; вне границ — значение на границе), столбцы стандартизованы, гребень λ = 250.
Суммы матриц — через einsum (matmul numpy 2.0.2 + Accelerate даёт ложные RuntimeWarning, см. `v4_c_fit.mv`)."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline
from scipy.special import expit

LAM = 250.0
HUBER = 1.345
IRLS_ITERS, IRLS_TOL = 100, 1e-10
CFGS = ["CLD1", "CLD2", "CLD3", "CLD4"]


def _mv(X, b):
    return np.einsum("ij,j->i", X, b)


def ridge_w(X: np.ndarray, y: np.ndarray, lam: float = LAM, w: np.ndarray | None = None) -> tuple:
    """Взвешенный гребень со свободным членом: (a, β); w=None — веса 1. Свободный член — взвешенное среднее, штраф только на β."""
    w = np.ones(len(y)) if w is None else w
    sw = w.sum()
    xm, ym = np.einsum("i,ij->j", w, X) / sw, float(np.einsum("i,i->", w, y)) / sw
    Xc = X - xm
    G = np.einsum("ij,i,ik->jk", Xc, w, Xc) + lam * np.eye(X.shape[1])
    b = np.linalg.solve(G, np.einsum("ij,i->j", Xc, w * (y - ym)))
    return ym - float(np.einsum("j,j->", xm, b)), b


def huber(X: np.ndarray, y: np.ndarray, delta: float, lam: float = LAM) -> tuple:
    """IRLS для потерь Хьюбера с гребнем: -> (a, β, число итераций, сошлась ли)."""
    a, b = ridge_w(X, y, lam)
    for it in range(1, IRLS_ITERS + 1):
        r = np.abs(y - a - _mv(X, b))
        w = np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-300))
        a2, b2 = ridge_w(X, y, lam, w)
        done = max(abs(a2 - a), float(np.abs(b2 - b).max(initial=0.0))) < IRLS_TOL
        a, b = a2, b2
        if done:
            return a, b, it, True
    return a, b, IRLS_ITERS, False


def bspline_cols(z: np.ndarray, knots_in: float, lo: float, hi: float) -> np.ndarray:
    """4 столбца кубического B-сплайна (5 базисных функций без первой — она заменяет свободный член). z вне [lo, hi] обрезается."""
    if hi - lo < 1e-12:
        return np.zeros((len(z), 4))
    t = np.concatenate([[lo] * 4, [min(max(knots_in, lo + 0.05 * (hi - lo)), hi - 0.05 * (hi - lo))], [hi] * 4])
    return BSpline.design_matrix(np.clip(z, lo, hi), t, 3).toarray()[:, 1:]


class Spline:
    """Базис по окну оценки: узлы и стандартизация столбцов фиксируются на нём; `transform` применяется к строкам прогноза."""

    def fit(self, Z: np.ndarray) -> "Spline":
        self.lo, self.hi, self.med = Z.min(0), Z.max(0), np.median(Z, 0)
        B = self._raw(Z)
        self.mu, sd = B.mean(0), B.std(0)
        self.sd = np.where(sd > 1e-12, sd, 1.0)
        return self

    def _raw(self, Z):
        return np.hstack([bspline_cols(Z[:, j], self.med[j], self.lo[j], self.hi[j]) for j in range(Z.shape[1])])

    def transform(self, Z):
        return (self._raw(Z) - self.mu) / self.sd


def fit_model(cfg: str, Z: np.ndarray, y: np.ndarray) -> dict:
    """Окно оценки: Z (N, k) стандартизованная, y = D. -> {predict(Zq), s (ŝ), c (p90 |D|), coef (по признакам), ok}."""
    if cfg not in CFGS:
        raise ValueError("неизвестная конфигурация CLD: %s" % cfg)
    c, k = float(np.quantile(np.abs(y), 0.9)), Z.shape[1]
    if cfg == "CLD4":
        sp = Spline().fit(Z)
        B = sp.transform(Z)
        a, b = ridge_w(B, y)
        res = y - a - _mv(B, b)
        amp = [float(np.std(_mv(B[:, 4 * j:4 * j + 4], b[4 * j:4 * j + 4]))) for j in range(k)]
        return {"predict": lambda Zq: a + _mv(sp.transform(Zq), b), "s": float(res.std()), "c": c, "coef": amp, "ok": True}
    a, b = ridge_w(Z, y)
    s_ridge = float((y - a - _mv(Z, b)).std())
    ok = True
    if cfg == "CLD3":
        a, b, _, ok = huber(Z, y, HUBER * s_ridge)
    s = float((y - a - _mv(Z, b)).std())
    return {"predict": lambda Zq: a + _mv(Zq, b), "s": s, "c": c, "coef": [float(x) for x in b], "ok": ok}


def weight(cfg: str, dhat: np.ndarray, s: float, c: float) -> np.ndarray:
    """Вес на M: w = σ(D̂/ŝ) (CLD1, 3, 4); CLD2 — clip(0,5 + D̂/(2c), 0, 1). D̂ ≡ 0 → w = 0,5."""
    if cfg == "CLD2":
        return np.clip(0.5 + dhat / (2.0 * c), 0.0, 1.0) if c > 0 else np.full(len(dhat), 0.5)
    return expit(dhat / s) if s > 0 else np.full(len(dhat), 0.5)
