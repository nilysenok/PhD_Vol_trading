"""S-C-DATA: PIT-фактор из 1–2 главных компонент доходностей M10 бумаг пула (без SBER).

Нагрузки и нормировка (σ_i, корреляции) оцениваются ТОЛЬКО по барам окна W прошлых сессий; для сессии s окно — сессии s−LAG−W+1 … s−LAG
(LAG = 1: сессия s не входит). Окно W = 120 сессий (~6300 баров): ошибка выборочной корреляции ~1/√n ≈ 0,013, при этом нагрузки
успевают следовать за режимом; 60 сессий дают вдвое больший шум в PC2. Пропуски не заполняются: σ_i и корреляции — по парно присутствующим барам;
бумага входит в окно, если у неё ≥ 50 % баров окна; фактор бара — МНК-оценка по присутствующим бумагам (≥ 8), не среднее с константой.
Знак и порядок PC между сессиями произвольны; для проекции важно только линейное пространство факторов сессии.
"""
from __future__ import annotations

import numpy as np

W_SESS = 120        # окно нагрузок, сессий
MIN_HIST = 60       # меньше прошлых сессий — нагрузок нет
LAG = 1             # окно заканчивается за LAG сессий до s (LAG = 1: строго прошлое)
MIN_PAPERS = 8      # бумаг с баром для оценки фактора
MIN_FRAC = 0.5      # доля баров окна, при которой бумага входит в нагрузки
MIN_OVERLAP = 30    # парных баров для корреляции пары (иначе 0)
LAM_FRAC = 0.05     # нижняя граница обусловленности МНК бара


def loadings(Rw: np.ndarray, k: int = 2):
    """Rw (n, P) доходности окна (NaN — нет) -> (sigma (P,) с NaN у исключённых, W (P, k)) или None."""
    valid = np.isfinite(Rw)
    cnt = valid.sum(0)
    sig = np.sqrt(np.square(np.where(valid, Rw, 0.0)).sum(0) / np.maximum(cnt, 1))
    inc = (cnt >= MIN_FRAC * len(Rw)) & (sig > 0)
    if inc.sum() < MIN_PAPERS:
        return None
    Z = np.where(valid, Rw / np.where(sig > 0, sig, 1.0), 0.0)[:, inc]
    M = valid[:, inc].astype(float)
    ov = M.T @ M
    C = np.where(ov >= MIN_OVERLAP, (Z.T @ Z) / np.maximum(ov, 1), 0.0)
    np.fill_diagonal(C, 1.0)
    ev, V = np.linalg.eigh(C)
    Wf = np.zeros((Rw.shape[1], k))
    Wf[inc] = V[:, ::-1][:, :k]
    return np.where(inc, sig, np.nan), Wf


def scores(R: np.ndarray, sig: np.ndarray, Wf: np.ndarray) -> np.ndarray:
    """Фактор бара: МНК z_i ≈ Σ_j f_j W_ij по присутствующим бумагам. -> F (n, k), NaN — бар не покрыт."""
    k = Wf.shape[1]
    z = R / sig
    valid = np.isfinite(z)
    M = valid.astype(float)
    z0 = np.where(valid, z, 0.0)
    A = np.einsum("np,pk,pl->nkl", M, Wf, Wf)
    cnt = valid.sum(1)
    n_inc = max(int((Wf[:, 0] != 0).sum()), 1)
    good = cnt >= MIN_PAPERS
    lam = np.linalg.eigvalsh(A)[:, 0]
    good &= lam >= LAM_FRAC * cnt / n_inc
    F = np.full((len(R), k), np.nan)
    if good.any():
        F[good] = np.linalg.solve(A[good], np.einsum("np,pk->nk", z0[good], Wf)[:, :, None])[:, :, 0]
    return F


def window(off: np.ndarray, s: int) -> tuple:
    """Границы баров окна нагрузок сессии s (прошлые сессии) и число сессий в нём."""
    e = s - LAG + 1
    lo = max(0, e - W_SESS)
    return int(off[lo]), int(off[max(e, lo)]), max(e - lo, 0)
