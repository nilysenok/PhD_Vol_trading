"""D2: круговой блочный бутстрап (блок 20, 5000, seed 42), MCS (T_max, Hansen–Lunde–Nason), бутстрап разности QLIKE, Холм с N, DM по периодам."""
from __future__ import annotations

import numpy as np

from models_v2.s3 import dm as dmod
from models_v2.s5.b1 import evalrows

BLOCK, B, SEED, ALPHA = 20, 5000, 42, 0.10
PERIODS = (("all", "весь период"), ("p2017_2021", "2017–2021"), ("y2022", "2022"), ("p2023_2025", "2023–2025"), ("no_240222", "без 24.02.2022"))


def block_idx(n: int, B_: int = B, block: int = BLOCK, seed: int = SEED, shuffle_dates: bool = False) -> np.ndarray:
    """(B, n) индексы: смежные блоки по block дат (по кругу), старты равномерно. shuffle_dates=True — мутация MD2 (блоки из перемешанных дат)."""
    rng = np.random.default_rng(seed)
    nb = -(-n // block)
    st = rng.integers(0, n, size=(B_, nb))
    idx = ((st[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(B_, nb * block)[:, :n]
    if shuffle_dates:
        idx = np.random.default_rng(seed + 1).permutation(n)[idx]
    return idx.astype(np.int32)


def boot_means(L: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """L (n,) или (n, M) -> (B,) или (B, M) средние по бутстрап-выборкам."""
    if L.ndim == 1:
        return L[idx].mean(axis=1)
    return np.column_stack([L[:, j][idx].mean(axis=1) for j in range(L.shape[1])])


def boot_delta(dl: np.ndarray, idx: np.ndarray) -> dict:
    """dl = QLIKE(конфиг) − QLIKE(база) по датам. P* = (k+1)/(B+1), k — повторов со средним Δ* >= 0 (односторонний)."""
    bm = boot_means(np.asarray(dl, float), idx)
    return {"mean": float(np.mean(dl)), "lo95": float(np.percentile(bm, 2.5)), "hi95": float(np.percentile(bm, 97.5)),
            "p_boot": float((np.sum(bm >= 0) + 1) / (len(bm) + 1))}


def mcs(L: np.ndarray, idx: np.ndarray, alpha: float = ALPHA) -> dict:
    """L (n, M) — потери. -> {'p': (M,) MCS p-значения, 'in': (M,) bool (p >= alpha), 'order': порядок исключения}."""
    M = L.shape[1]
    mu, bm = L.mean(axis=0), boot_means(L, idx)
    alive, pm, order, run = list(range(M)), np.ones(M), [], 0.0
    while True:
        A = np.array(alive)
        if len(A) == 1:
            pm[A] = 1.0   # последняя модель — p = 1 по определению
            break
        d_i = mu[A] - mu[A].mean()
        b_i = bm[:, A] - bm[:, A].mean(axis=1, keepdims=True)
        sd = b_i.std(axis=0, ddof=1)
        ok = sd > 1e-15
        t = np.where(ok, d_i / np.where(ok, sd, 1.0), 0.0)
        tb = np.where(ok, (b_i - d_i) / np.where(ok, sd, 1.0), 0.0).max(axis=1)
        p = float(np.mean(tb >= t.max() - 1e-12))
        run = max(run, p)
        if p >= alpha:
            pm[A] = run
            break
        j = int(A[int(np.argmax(t))])
        pm[j] = run
        order.append(j)
        alive.remove(j)
    return {"p": pm, "in": pm >= alpha, "order": order}


def holm_n(p, N: int) -> np.ndarray:
    """Холм с N сравнениями: недостающие до N p = 1 (консервативно). N < числа тестов — ошибка."""
    p = np.asarray(p, float)
    if N < len(p):
        raise ValueError("N=%d меньше числа тестов %d" % (N, len(p)))
    return dmod.holm(np.concatenate([p, np.ones(N - len(p))]))[:len(p)]


def dm_periods(rv, s2_base, s2_cfg, td) -> list:
    """DM (Ньюи–Уэст лаг 5) cfg против base по периодам: [{'period', 'd', 'dm', 'p', 'n'}]; d < 0 — конфиг лучше."""
    mk, out = evalrows.masks(td), []
    for key, lab in PERIODS:
        r = evalrows.vs(np.asarray(rv, float), np.asarray(s2_base, float), np.asarray(s2_cfg, float), mk[key])
        out.append({"period": lab, **r, "n": int(mk[key].sum())})
    return out
