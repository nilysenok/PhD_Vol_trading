"""V5 лист 1a: связи между бумагами (план п. 3.1): матрица W (вариант C, GS4 — три соседа) и вектор λ (вариант D, GS5 — низкий ранг) по ПРОШЛЫМ s̃.
Правила (объявлены до счёта): пара с < 120 общих дней (`MIN_COMMON`) делает строку равновесной (T = среднее s̃ остальных); соседи — по положительной корреляции, веса ∝ корреляции, сумма 1; λ — первый собственный вектор прошлой корреляции s̃ (знак: среднее > 0),
λ̃ = ½λ + ½λ̄. LOTO: исключённая бумага не участвует в оценке и не питает остальных; её собственная строка — равные веса, c = 1 (λ_i = λ̄)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s12.v5_gs_core import equal_weights

MIN_COMMON = 120
TOP = 3


def _corr(X: np.ndarray, keep: list) -> np.ndarray:
    return pd.DataFrame(X[:, keep]).corr(min_periods=MIN_COMMON).to_numpy()


def graph_c(X: np.ndarray, excl: int | None) -> tuple:
    """GS4: W (строка i: три соседа с наибольшей положительной корреляцией, веса ∝ ей). -> (W, c, число равновесных строк, средний наибольший вес)."""
    m = X.shape[1]
    W, c = equal_weights(m)
    keep = [j for j in range(m) if j != excl]
    C = _corr(X, keep)
    n_eq, mx = 0, []
    for a, i in enumerate(keep):
        r = np.delete(C[a], a)
        oth = [j for j in keep if j != i]
        pos = np.flatnonzero(np.nan_to_num(r, nan=-1.0) > 0)
        if np.isnan(r).any() or len(pos) == 0:
            n_eq += 1
            continue
        top = pos[np.argsort(-r[pos], kind="stable")[:TOP]]
        W[i, :] = 0.0
        W[i, [oth[t] for t in top]] = r[top] / r[top].sum()
        mx.append(W[i].max())
    if excl is not None:
        W[:, excl] = 0.0
    return W, c, n_eq, (float(np.mean(mx)) if mx else float("nan"))


def graph_d(X: np.ndarray, excl: int | None) -> tuple:
    """GS5: вес j в сумме = λ̃_j, множитель строки c_i = λ̃_i/λ̄. Недостаточно данных / знаки не положительны — равные веса. -> (W, c, n_eq (0 или m), отношение max c / min c)."""
    m = X.shape[1]
    W, c = equal_weights(m)
    keep = [j for j in range(m) if j != excl]
    C = _corr(X, keep)
    if np.isnan(C).any():
        return W, c, len(keep), float("nan")
    v = np.linalg.eigh(C)[1][:, -1]
    v = v if v.sum() > 0 else -v
    lb = float(v.mean())
    lt = 0.5 * v + 0.5 * lb
    if lb <= 0 or (lt <= 0).any():
        return W, c, len(keep), float("nan")
    for a, i in enumerate(keep):
        W[i, :] = 0.0
        W[i, keep] = lt
        W[i, i] = 0.0
        c[i] = lt[a] / lb
    if excl is not None:
        W[:, excl] = 0.0
    return W, c, 0, float(c[keep].max() / c[keep].min())


def graph(var: str, X: np.ndarray, excl: int | None = None) -> tuple:
    """-> (W, c, n_eq, сводка). A, B — равные веса (GS1/GS2 при β = 0 передачи нет; GS3 — рыночный score)."""
    m = X.shape[1]
    if var == "C":
        return graph_c(X, excl)
    if var == "D":
        return graph_d(X, excl)
    W, c = equal_weights(m)
    if excl is not None:
        W[:, excl] = 0.0
    return W, c, 0, float("nan")
