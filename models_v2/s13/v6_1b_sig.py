"""V6-1b, G17: признаки подписи траектории дня (план V6, п. 3). Траектория X_k = (τ, P, V, N, Q)_k, X_0 = 0, по барам основной сессии: τ = k/n; P = Σ r/√RV_d; V = Σ r²/RV_d; N = Σ r²·1{r<0}/RV_d; Q = Σ vol/Vol_d.
Площади Леви A_ij = ½ Σ_{k<l}(ΔX_i,k ΔX_j,l − ΔX_j,k ΔX_i,l); Σ = шесть площадей без τ (PV, PN, PQ, VN, VQ, NQ) за день d и их средние за 5 сессий ≤ d (12 признаков).
Σ³ = логарифм подписи 3-го порядка по (τ, P, V) в базисе Линдона (8 слов длины 3 над 3 буквами), за день d. Подпись считается своим кодом (без внешних пакетов): прямой формулой итерированных интегралов кусочно-линейного пути.
Канал N или Q дня равен 0, если RS⁻ = 0 (N ≡ 0) или Vol = 0 (Q ≡ 0): считаются в `flags`. Бары — те же, что у S (`hv2_m10.bar_returns`, основная сессия)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from models_v2 import daily as D
from models_v2.s9 import hv2_m10 as M

CH = ("tau", "P", "V", "N", "Q")
PAIRS = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
AREA_NAMES = ["a_" + CH[i] + CH[j] for i, j in PAIRS]
M5_NAMES = ["m5_" + n for n in AREA_NAMES]
SIGMA_NAMES = AREA_NAMES + M5_NAMES
WIN5 = 5
LAST = 3                                   # Σ³ — по первым трём каналам (τ, P, V)


def increments(r, vol) -> tuple:
    """Приращения каналов одного дня (n × 5) и флаги (RS⁻ = 0, Vol = 0). RV дня > 0 обязателен (канонические дни)."""
    r, vol = np.asarray(r, float), np.asarray(vol, float)
    q, tot = r * r, float(np.sum(vol))
    rv = float(q.sum())
    inc = np.column_stack([np.full(len(r), 1.0 / len(r)), r / np.sqrt(rv), q / rv, np.where(r < 0, q, 0.0) / rv, vol / tot if tot > 0 else np.zeros(len(r))])
    return inc, bool(not (r < 0).any()), bool(tot <= 0)


def excl_cumsum(a) -> np.ndarray:
    return np.cumsum(a, axis=0) - a


def areas(inc) -> np.ndarray:
    """Антисимметричная матрица площадей Леви (k × k): A_ij = ½ Σ_l (C_i,<l a_j,l − C_j,<l a_i,l), C — накопленная сумма приращений до шага l."""
    c = excl_cumsum(inc)
    s = np.einsum("li,lj->ij", c, inc)
    return 0.5 * (s - s.T)


def sig3(inc) -> tuple:
    """Подпись пути до уровня 3 (S1, S2, S3) прямой формулой для кусочно-линейного пути: S2_ij = Σ_{a<b} a_i b_j + ½ Σ a_i a_j; S3 — по сегменту, где лежит третье время."""
    a = np.asarray(inc, float)
    c = excl_cumsum(a)
    seg2 = np.einsum("li,lj->lij", c, a) + 0.5 * np.einsum("li,lj->lij", a, a)
    p2 = excl_cumsum(seg2)
    s2 = seg2.sum(0)
    s3 = np.einsum("lij,lk->ijk", p2, a) + 0.5 * np.einsum("li,ljk->ijk", c, np.einsum("lj,lk->ljk", a, a)) + np.einsum("li,lj,lk->ijk", a, a, a) / 6.0
    return a.sum(0), s2, s3


def log3(inc) -> tuple:
    """Логарифм подписи: (L1, L2, L3) = (S1, S2 − ½ S1⊗S1, S3 − ½(S1⊗S2 + S2⊗S1) + ⅓ S1⊗S1⊗S1)."""
    s1, s2, s3 = sig3(inc)
    l2 = s2 - 0.5 * np.outer(s1, s1)
    l3 = s3 - 0.5 * (np.einsum("i,jk->ijk", s1, s2) + np.einsum("ij,k->ijk", s2, s1)) + np.einsum("i,j,k->ijk", s1, s1, s1) / 3.0
    return s1, l2, l3


def is_lyndon(w: tuple) -> bool:
    return all(w < w[i:] + w[:i] for i in range(1, len(w)))


def lyndon_words(k: int, n: int) -> list:
    return [w for w in itertools.product(range(k), repeat=n) if is_lyndon(w)]


def _bracket(a: dict, b: dict) -> dict:
    out: dict = {}
    for (u, x), (v, y) in itertools.product(a.items(), b.items()):
        out[u + v] = out.get(u + v, 0.0) + x * y
        out[v + u] = out.get(v + u, 0.0) - x * y
    return out


def lyndon_poly(w: tuple) -> dict:
    """Базисный элемент P_w = [P_u, P_v] по стандартной факторизации (v — самый длинный собственный суффикс-слово Линдона)."""
    if len(w) == 1:
        return {w: 1.0}
    i = next(i for i in range(1, len(w)) if is_lyndon(w[i:]))
    return _bracket(lyndon_poly(w[:i]), lyndon_poly(w[i:]))


def lyndon_matrix(k: int = LAST, n: int = 3) -> tuple:
    """-> (слова Линдона длины n, матрица B (k^n × число слов): столбец j — разложение P_{w_j} по словам)."""
    ws = lyndon_words(k, n)
    idx = {w: i for i, w in enumerate(itertools.product(range(k), repeat=n))}
    B = np.zeros((k ** n, len(ws)))
    for j, w in enumerate(ws):
        for u, x in lyndon_poly(w).items():
            B[idx[u], j] = x
    return ws, B


def lyndon_coords(l3: np.ndarray) -> tuple:
    """Координаты L3 в базисе Линдона: B c = vec(L3); возвращает (c, невязка) — невязка ≈ 0 для настоящего логарифма подписи (элемент алгебры Ли)."""
    ws, B = lyndon_matrix(l3.shape[0], 3)
    v = l3.reshape(-1)
    c = np.linalg.lstsq(B, v, rcond=None)[0]
    return c, float(np.abs(B @ c - v).max())


def sigma3_names() -> list:
    return ["sig3_" + "".join(CH[i] for i in w) for w in lyndon_words(LAST, 3)]


def day_values(r, vol) -> tuple:
    """-> (6 площадей, 8 координат Σ³, флаг RS⁻ = 0, флаг Vol = 0) одного дня."""
    inc, f_rs, f_vol = increments(r, vol)
    A = areas(inc)
    c, _ = lyndon_coords(log3(inc[:, :LAST])[2])
    return np.array([A[i, j] for i, j in PAIRS]), c, f_rs, f_vol


def day_table(bars: pd.DataFrame) -> pd.DataFrame:
    """Бары -> по сессиям: 6 площадей, 8 координат Σ³, флаги. Только бары самого дня. МУТАНТ «бары дня t⁺ в признаке дня d» сдвигает результат в тесте."""
    main = D.main_session(bars).sort_values("begin").reset_index(drop=True)
    r = M.bar_returns(bars)
    vol = main["volume"].to_numpy(float)
    rows = {}
    for d, ix in r.groupby("date").indices.items():
        a, c, f_rs, f_vol = day_values(r["r"].to_numpy(float)[ix], vol[ix])
        rows[d] = np.r_[a, c, float(f_rs), float(f_vol)]
    cols = AREA_NAMES + sigma3_names() + ["flag_rs0", "flag_vol0"]
    return pd.DataFrame.from_dict(rows, orient="index", columns=cols).rename_axis("date").sort_index()


def five_mean(x: pd.DataFrame, win: int = WIN5) -> pd.DataFrame:
    """Среднее за `win` сессий ≤ d (окно назад, включая d). МУТАНТ «с днём d+1» подменяет эту функцию в тесте."""
    return x.rolling(win, min_periods=win).mean()


def features(bars: pd.DataFrame, daily: pd.DataFrame) -> tuple:
    """-> (Σ: 12 колонок, Σ³: 8 колонок, флаги) на днях панели `daily.index`."""
    t = day_table(bars).reindex(daily.index)
    a = t[AREA_NAMES]
    sigma = pd.concat([a, five_mean(a).set_axis(M5_NAMES, axis=1)], axis=1)
    return sigma, t[sigma3_names()], t[["flag_rs0", "flag_vol0"]]
