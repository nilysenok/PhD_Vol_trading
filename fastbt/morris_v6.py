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
# Оси-допуски (Ж-142 п. 2, решение 30.10 16:00): контекстные фильтры, перемножаемые по «И» (eval_v6: KONTEKST в _ok6, FILTR6, sob_tip, vn_uslovie,
# пауза) — запрещающие, `filtr: true` в кубах v6 (сверяет tests/test_morris_baza.py). База траектории — все они «выкл.» (индекс 0).
OSI_DOPUSK = ("hurst_th", "naklon", "vol_regime_max", "tihiy_othod", "sektor_trend", "shirina", "relf_napr", "sob_tip", "post_shok",
              "szhatie", "rv_perevorot", "obyom_anom", "rezhim", "vn_uslovie", "pauza_porog")
# Фильтры v5 (раздел `filtr` кубов v5, 0 = «выкл.»): тоже «И»; в базе траектории тоже «выкл.» (толкование 30.10 17:40, расследование dopuski §0).
OSI_FILTR_V5 = ("adx_th", "atr_lo", "vol_th", "rsi_th", "mom_n", "dow", "squeeze_pct")
# Оси, значения которых включают требование ранга (eval_v5_osi.razbor: r.rang): m_sily ≠ "1" (lin, kvadr), bezubytok = rang:*, tejk = sila:*.
# Ранг у редкого сигнала NaN (MIN_NABL = 20 прошлых кандидатов) → кандидат не входит / вес 0: база с рангом даёт Y = 0 (расследование s6nol, 30.10).
# Первое значение всех трёх — без ранга ("1", "net", "net"); в базе траектории они «не требует ранга» (индекс 0), ранг включается на своём шаге.
OSI_RANG = ("m_sily", "bezubytok", "tejk")
OSI_BAZA = OSI_DOPUSK + OSI_FILTR_V5 + OSI_RANG


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


def traektorii_kanon(L, r: int, klyuch, seed: int = SEED, mnozh: int = MNOZH, popytok: int = 500, baza=()) -> tuple[np.ndarray, np.ndarray]:
    """Траектории только по каноническим точкам: klyuch(индексы tuple) → ключ канона или None (точка недопустима).

    Правило (детерминировано seed): отбор Кампалонго — как в `traektorii`, затем починка каждой траектории по порядку осей.
    Старт None → заменяется случайной допустимой точкой (до `popytok` попыток, иначе ValueError).
    Шаг оси a из текущей точки x: кандидаты — предложенный индекс (из исходной траектории), затем остальные индексы оси в порядке
    перестановки seed. Берётся первый кандидат с каноном не None и ключом ≠ ключа x (активный шаг); нет такого — предложенный,
    если канонический (шаг неактивен: ключ тот же); иначе ось в этой траектории пропускается (точка повторяется, шаг не активен,
    эффект NaN). Шаг на альтернативный уровень может быть не на 2/3, в единицах Шарпа это не важно (на Δ не делим).
    baza — номера осей-допуски: старт каждой траектории с ними в «выкл.» (индекс 0), остальные оси случайны; допуск включается на своём шаге."""
    # xtask: длинная — на 4 строки docstring больше лимита; починка траектории — один проход со общим кэшем ключей
    I, por = traektorii(L, r, seed, mnozh)
    Ln, rng, kl, b = np.asarray(L), np.random.default_rng([seed, 1]), {}, np.asarray(baza, dtype=np.int64)

    def key(x):
        t = tuple(int(i) for i in x)
        if t not in kl:
            kl[t] = klyuch(t)
        return kl[t]

    for m in range(r):
        x = I[m, 0].astype(np.int64)
        x[b] = 0
        for _ in range(popytok if key(x) is None else 0):
            x = rng.integers(0, Ln)
            x[b] = 0
            if key(x) is not None:
                break
        if key(x) is None:
            raise ValueError(f"траектория {m}: нет допустимой стартовой точки за {popytok} попыток")
        I[m, 0], pred = x, I[m].copy()
        for t in range(1, I.shape[1]):
            a, kx, zapas = por[m, t - 1], key(x), None
            ost = [v for v in rng.permutation(Ln[a]) if v not in (x[a], pred[t, a])]
            for v in ([int(pred[t, a])] if pred[t, a] != x[a] else []) + ost:
                y = x.copy()
                y[a] = v
                ky = key(y)
                if ky is None:
                    continue
                if ky != kx:
                    x = y
                    break
                zapas = y if zapas is None else zapas
            else:
                x = x if zapas is None else zapas
            I[m, t] = x
    return I, por


def pary_nol(keys: np.ndarray, poryadok: np.ndarray, nol: np.ndarray) -> np.ndarray:
    """Число активных шагов по осям, где обе точки без сделок (Д4): их разность 0 — не доказательство «пустоты» оси."""
    nol = np.asarray(nol, dtype=bool)
    n = np.zeros(poryadok.shape[1], dtype=np.int64)
    np.add.at(n, poryadok, ((keys[:, 1:] != keys[:, :-1]) & nol[:, 1:] & nol[:, :-1]).astype(np.int64))
    return n


def effekty(Y: np.ndarray, keys: np.ndarray, poryadok: np.ndarray, nol=None) -> np.ndarray:
    """Элементарные эффекты (r, k): ΔY шага, меняющего ось; NaN, где шаг не активен (ключ тот же), нет Y у одной из точек
    или (nol — bool (r, k+1), точка без сделок) обе точки пары без сделок."""
    Y = np.asarray(Y, dtype=np.float64)
    dY = Y[:, 1:] - Y[:, :-1]
    ok = (keys[:, 1:] != keys[:, :-1]) & np.isfinite(dY)
    if nol is not None:
        nol = np.asarray(nol, dtype=bool)
        ok &= ~(nol[:, 1:] & nol[:, :-1])
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
