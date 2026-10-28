# -*- coding: utf-8 -*-
"""progon_region.py — Л-7: регион и плато (прогон этапа 1). Чистые функции над матрицей и строками ledger; данных не читает.

M — дни × столбцы (Chitatel.matrica); rows[j] — строка ledger столбца j (точка: row["pt"], или row["klyuch"]["pt"], или
row["klyuch"]). Порядок: ось -> полный индекс Соболя (Jansen) целевой на переданных кварталах -> K_OSI ведущих осей ->
регионы радиуса RADIUS по этим осям (по остальным — все значения) -> допустимые -> лучший по целевой ряда региона.
Ансамбль региона — среднее рядов членов (равные доли капитала). Уровни оси упорядочены: числа по возрастанию, затем токены.
Отступление: Jansen считается по парам «ближайший в остальных осях» (пул ≤ POOL строк), а не по pick-freeze Соболя.
"""
from __future__ import annotations

import math

import numpy as np

from fastbt.pbo_pravilo import sigma126

K_OSI, RADIUS, DOLYA_MIN, MIN_CHLENOV = 4, 1, 0.8, 10
TIE, MIN_SDELOK, NK, POOL, NS, CHUNK = 0.05, 30, 16, 3000, 2000, 256


def _pt(row: dict) -> dict:
    if "pt" in row:
        return row["pt"]
    kl = row["klyuch"]
    return kl.get("pt", kl)


def _lvl(v):
    return (0, float(v), "") if isinstance(v, (int, float)) and not isinstance(v, bool) else (1, 0.0, str(v))


def osi_rows(rows: list) -> list:
    """Оси, у которых в rows больше одного значения (имена по алфавиту)."""
    pts = [_pt(r) for r in rows]
    names = sorted({k for p in pts for k in p})
    return [a for a in names if len({str(p.get(a)) for p in pts}) > 1]


def kodirovat(rows: list, osi: list) -> np.ndarray:
    """n × len(osi): индекс уровня каждой оси."""
    pts = [_pt(r) for r in rows]
    X = np.zeros((len(rows), len(osi)), dtype=np.int16)
    for c, a in enumerate(osi):
        lv = sorted({p.get(a) for p in pts}, key=_lvl)
        look = {v: i for i, v in enumerate(lv)}
        X[:, c] = [look[p.get(a)] for p in pts]
    return X


def rang_osey(rows, tsel, osi) -> list:
    """Полный индекс Соболя (Jansen) целевой tsel (длина len(rows)) по каждой оси из osi: E[(Y−Y′)²]/(2 Var Y), где Y′ —
    ближайшая по остальным осям строка с другим значением этой оси (пул ≤ POOL строк, ядро детерминировано)."""
    y = np.asarray(tsel, float)
    X = kodirovat(rows, osi).astype(np.int8)
    ok = np.flatnonzero(np.isfinite(y))
    if len(ok) < 3 or not osi:
        return [0.0] * len(osi)
    rng = np.random.default_rng(0)
    pool = np.sort(rng.choice(ok, min(POOL, len(ok)), replace=False))
    src = np.sort(rng.choice(ok, min(NS, len(ok)), replace=False))
    var = float(np.var(y[ok], ddof=1))
    sums, cnt = np.zeros(len(osi)), np.zeros(len(osi))
    for s0 in range(0, len(src), CHUNK):
        s = src[s0:s0 + CHUNK]
        D = X[s][:, None, :] != X[pool][None, :, :]
        tot = D.sum(axis=2)
        for i in range(len(osi)):
            valid = D[:, :, i]
            j = np.argmin(np.where(valid, tot - valid, 99), axis=1)
            has = valid.any(axis=1)
            sums[i] += ((y[s] - y[pool[j]])[has] ** 2).sum()
            cnt[i] += has.sum()
    return [float(sums[i] / cnt[i] / (2 * var)) if cnt[i] and var > 0 else 0.0 for i in range(len(osi))]


def regiony(rows, osi_top, radius) -> dict:
    """Центр (кортеж индексов по osi_top, занятая клетка) -> индексы строк с |Δиндекса| ≤ radius по каждой из osi_top."""
    X = kodirovat(rows, osi_top)
    out = {}
    for c in np.unique(X, axis=0):
        out[tuple(int(v) for v in c)] = np.flatnonzero((np.abs(X - c) <= radius).all(axis=1))
    return out


def ryad_regiona(M: np.ndarray, chleny) -> np.ndarray:
    """Ансамбль: среднее столбцов членов (равные доли капитала)."""
    return np.asarray(M)[:, list(chleny)].mean(axis=1)


def _kv_dnei(daty) -> np.ndarray:
    m = np.asarray(daty).astype("datetime64[M]").astype(int)
    return (m // 12 + 1970 - 2016) * 4 + (m % 12) // 3


def _kv_sr(M: np.ndarray, daty, kv: list) -> np.ndarray:
    """N × len(kv): Шарп квартала (годовая шкала, ddof=1); σ = 0 -> 0; меньше 2 дней -> nan."""
    k = _kv_dnei(daty)
    out = np.full((M.shape[1], len(kv)), np.nan)
    for c, q in enumerate(kv):
        X = M[k == q]
        if len(X) < 2:
            continue
        sd = X.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[:, c] = np.where(sd > 0, X.mean(axis=0) / np.where(sd > 0, sd, 1.0) * math.sqrt(252), 0.0)
    return out


def _dolya_pol(sr) -> float:
    """Доля кварталов с SR > 0; nan — не положительный."""
    return float(np.mean(np.asarray(sr, float) > 0))


def _cel_stolbcov(M, daty, kv, tsel) -> np.ndarray:
    """Целевая каждого столбца по кварталам kv: «mediana» (nan, если есть nan) | «dolya»."""
    sr = _kv_sr(M, daty, kv)
    if tsel == "dolya":
        return (sr > 0).mean(axis=1)
    return np.where(np.isnan(sr).any(axis=1), np.nan, np.median(np.nan_to_num(sr), axis=1))


def _aktivna(row: dict, base) -> bool:
    sd = row.get("sdelok_y") or {}
    st = row.get("storony_y") or {y: 2 * v for y, v in sd.items()}
    if not sd or not all(v >= MIN_SDELOK for v in sd.values()):
        return False
    return base is None or all(v >= 0.5 * base for v in st.values())


def vybor_region(M, rows, daty, kvartaly_idx=None, radius=RADIUS, tsel="mediana", *, dop=None, storony_yakor=None):
    """Лучший допустимый регион на кварталах kvartaly_idx (None — все 16) или None. Допустимость: членов ≥ MIN_CHLENOV,
    доля кварталов с SR ряда > 0 ≥ DOLYA_MIN, в центре есть строка, допустимая по активности (dop[j] или как dopustima)."""
    M, daty = np.asarray(M, float), np.asarray(daty).astype("datetime64[D]")
    kv = list(range(NK)) if kvartaly_idx is None else sorted(kvartaly_idx)
    mask = np.isin(_kv_dnei(daty), kv)
    Ms, ds = M[mask], daty[mask]
    tcol = _cel_stolbcov(M, daty, kv, tsel)
    osi_all = osi_rows(rows)
    st = rang_osey(rows, tcol, osi_all)
    osi = [osi_all[i] for i in np.argsort(-np.asarray(st), kind="stable")[:K_OSI]]
    X = kodirovat(rows, osi)
    act = np.asarray(dop, bool) if dop is not None else np.array([_aktivna(r, storony_yakor) for r in rows])
    cand = []
    for centr, idx in regiony(rows, osi, radius).items():
        if len(idx) < MIN_CHLENOV or not act[(X == np.array(centr)).all(axis=1)].any():
            continue
        ry = ryad_regiona(Ms, idx)
        if _dolya_pol(_kv_sr(ry[:, None], ds, kv)[0]) < DOLYA_MIN:
            continue
        sc = float(_cel_stolbcov(ry[:, None], ds, kv, tsel)[0])
        if math.isfinite(sc):
            cand.append((sc, centr, idx, ry))
    if not cand:
        return None
    top = max(c[0] for c in cand)
    near = [c for c in cand if c[0] >= top - TIE]
    sg = lambda c: (lambda s: s if math.isfinite(s) else math.inf)(sigma126(c[3], ds))  # noqa: E731
    sc, centr, idx, ry = min(near, key=sg)
    return dict(centr=centr, chleny=[int(i) for i in idx], ryad=ry, score=sc, osi=osi, tstolb=tcol)


def select_fn_region(rows, daty, coords=None):
    """Под pbo_pravilo.cscv: select_fn(M_in, coords, dates_in) -> индекс столбца | None. ❓ pbo_pravilo не принимает ряд:
    умолчание — член региона с лучшей целевой на train-половине (отступление). Кварталы — те, что есть в dates_in."""
    def fn(M_in, coords_in, dates_in):
        d = np.asarray(daty if dates_in is None else dates_in).astype("datetime64[D]")
        res = vybor_region(np.asarray(M_in, float).T, rows, d, sorted(set(_kv_dnei(d).tolist()) & set(range(NK))))
        if res is None:
            return None
        t = np.where(np.isfinite(res["tstolb"][res["chleny"]]), res["tstolb"][res["chleny"]], -np.inf)
        return int(res["chleny"][int(np.argmax(t))])
    return fn
