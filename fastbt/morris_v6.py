"""V6-14 (Ж-132, решение дирижёра 29.10 13:20): траектории Морриса, условный μ*, верхняя граница U95. Данных не читает, счёта не запускает.

План: docs/otchety/2026-10-29_setka_v6_inventarizaciya.md §3.3. Уровни p = 4, u ∈ {0, 1/3, 2/3, 1}, шаг Δ = 2/3 (два уровня):
у каждого уровня ровно один партнёр ((j + 2) mod 4), поэтому знак шага определяется стартом, а двоичная ось переключается
выкл. ↔ вкл. на каждом шаге. Индекс значения = round(u·(L − 1)), «выкл.» = 0 (в целых: (2j(L − 1) + 3) // 6, без плавающей точки).
Отбор — Кампалонго (2007): r лучших по разнесённости из 10·r кандидатов (жадное удаление, Ruano 2012; ничья — меньший номер).
Элементарный эффект = разность Шарпа за шаг (единицы Шарпа, на Δ не делится: порог 0,05 объявлен в единицах Шарпа).
Только активные шаги: ключ `canon_v6` точки изменился (ключ даёт вызывающий). Шаг, где у одной из точек нет Y (канон None, NaN), не считается.
Точка без сделок — Y = 0 с флагом, это решает вызывающий: здесь такая точка — обычное число.
"""
from __future__ import annotations

import numpy as np

SEED = 20261101
P, DELTA = 4, 2 / 3
MNOZH, B_BOOT = 10, 1000


def indeks(j, L):
    """Уровень j ∈ 0..3 (u = j/3) → индекс значения оси с L значениями: round(j/3·(L − 1)), 0 = «выкл.»."""
    return (2 * np.asarray(j, dtype=np.int64) * (np.asarray(L, dtype=np.int64) - 1) + 3) // 6


def kandidaty(k: int, m: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """m траекторий: уровни J (m, k+1, k) int8 и порядок осей (m, k); шаг t меняет ось poryadok[:, t-1] на ±2 уровня."""
    j0 = rng.integers(0, P, size=(m, k)).astype(np.int8)
    poryadok = np.argsort(rng.random((m, k)), axis=1)
    J = np.empty((m, k + 1, k), dtype=np.int8)
    J[:, 0] = j0
    ar = np.arange(m)
    for t in range(1, k + 1):
        J[:, t] = J[:, t - 1]
        a = poryadok[:, t - 1]
        J[ar, t, a] = (J[ar, t - 1, a] + 2) % P
    return J, poryadok


def razneseniye(J: np.ndarray) -> np.ndarray:
    """Квадраты расстояний Кампалонго между траекториями: d_ml² где d_ml = Σ_{точки i, j} ‖x_i^m − x_j^l‖. Диагональ обнулена."""
    m, n, k = J.shape
    u = J.astype(np.float64) / (P - 1)
    flat = u.reshape(-1, k)
    sq = (flat**2).sum(1)
    d = np.zeros((m, m))
    for i in range(m):
        with np.errstate(all="ignore"):  # numpy/Accelerate на macOS выставляет ложные флаги overflow в matmul; результат сверен в тесте
            d2 = sq[i * n:(i + 1) * n, None] + sq[None, :] - 2.0 * (u[i] @ flat.T)
        d[i] = np.sqrt(np.clip(d2, 0.0, None)).reshape(n, m, n).sum(axis=(0, 2))
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0.0)
    return d**2


def otbor(J: np.ndarray, r: int) -> np.ndarray:
    """Номера r траекторий с наибольшей суммой d² между собой: удаляем ту, чьё удаление теряет меньше всего."""
    d2 = razneseniye(J)
    zhivye = np.ones(len(J), dtype=bool)
    s = d2.sum(1)
    for _ in range(len(J) - r):
        c = np.where(zhivye, s, np.inf).argmin()
        zhivye[c] = False
        s -= d2[:, c]
    return np.flatnonzero(zhivye)


def traektorii(L, r: int, seed: int = SEED, mnozh: int = MNOZH) -> tuple[np.ndarray, np.ndarray]:
    """r траекторий по Кампалонго из mnozh·r кандидатов: индексы точек I (r, k+1, k) и порядок осей (r, k)."""
    L = np.asarray(L)
    J, poryadok = kandidaty(len(L), mnozh * r, np.random.default_rng(seed))
    keep = otbor(J, r) if mnozh > 1 else np.arange(r)
    return indeks(J[keep], L).astype(np.int16), poryadok[keep]


def effekty(Y: np.ndarray, keys: np.ndarray, poryadok: np.ndarray) -> np.ndarray:
    """Элементарные эффекты (r, k): ΔY шага, меняющего ось; NaN, где шаг не активен (ключ тот же) или нет Y у одной из точек."""
    Y = np.asarray(Y, dtype=np.float64)
    dY = Y[:, 1:] - Y[:, :-1]
    ok = (keys[:, 1:] != keys[:, :-1]) & np.isfinite(dY)
    dY = np.where(ok, dY, np.nan)
    E = np.full((len(Y), poryadok.shape[1]), np.nan)
    np.put_along_axis(E, poryadok, dY, axis=1)
    return E


def statistika(E: np.ndarray, B: int = B_BOOT, seed: int = SEED) -> dict[str, np.ndarray]:
    """По осям: n_akt, mu_star (среднее |EE| активных шагов), sigma (ddof=1, по EE со знаком), U95 (95-й процентиль бутстрапа по траекториям)."""
    akt = np.isfinite(E)
    n = akt.sum(0)
    A0 = np.where(akt, np.abs(E), 0.0)
    E0 = np.where(akt, E, 0.0)
    with np.errstate(all="ignore"):  # деление на n = 0 → NaN (ось без активных шагов); ложные флаги matmul — см. razneseniye
        mu = np.where(n > 0, A0.sum(0) / n, np.nan)
        dev = np.where(akt, E0 - E0.sum(0) / n, 0.0)
        sd = np.where(n > 1, np.sqrt((dev**2).sum(0) / (n - 1)), np.nan)
        w = np.random.default_rng(seed).multinomial(len(E), np.full(len(E), 1 / len(E)), size=B)
        mub = (w @ A0) / (w @ akt.astype(np.float64))
    u95 = np.full(E.shape[1], np.nan)
    for i in np.flatnonzero(np.isfinite(mub).any(0)):
        u95[i] = np.nanpercentile(mub[:, i], 95)
    return {"n_akt": n, "mu_star": mu, "sigma": sd, "U95": u95}
