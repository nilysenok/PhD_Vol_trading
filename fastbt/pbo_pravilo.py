# -*- coding: utf-8 -*-
"""pbo_pravilo.py — PBO по CSCV той же процедурой, что и отбор (Ж-080): in-sample выбор по правилу этапа, не argmax.

M — точки × дни (транспонированная матрица V.matrix). coords[j] — {ось: значение} точки j или None (не точка сетки:
конфигурация ансамбля; в матрице и в ранге остаётся, кандидатом центра плато не бывает). select_fn(M_in, coords,
dates_in) -> индекс точки | None (None: допустимой точки нет — разбиение пропущено и считается в n_propusk).

Правило vybor_centr_plato на половине (объявлено до счёта):
  * mSR по годам — из тех дней года, что попали в половину; год, в котором в половине < MIN_DNEY дней, в min не входит;
    целевая точки = min mSR по оставшимся годам (nan хотя бы в одном — nan), как V.obj;
  * плато, ≥ 3 конечных соседей (край), допустимость, топ-TOPK, ничья ≤ TIE по плато -> меньшая sigma126 — как pick_center;
  * sigma126 — по скользящим окнам внутри непрерывных кусков половины (окно не перескакивает пропущенный блок);
  * допустимость (dop) — по сделкам/сторонам за весь train: структурный фильтр активности, не доходность;
  * ворота G1…G5 внутри разбиения не применяются (G1 — сам результат; G3/G4 — отдельные доходные проверки).
Данных не читает.
"""
from __future__ import annotations

import functools
import itertools
import math
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd

TIE, WIN, TOPK, MIN_NEIGHBORS, MIN_DNEY, GAP_DNEY = 0.05, 126, 20, 3, 21, 20      # = V.TIE/WIN, etap1_poisk.TOPK (тест)
Z95 = -1.645                                                                       # = metriki_e.Z95 (тест)


class Kub(NamedTuple):
    """Плато-куб осей плана: порядок строк = itertools.product (как у factorial_sweep)."""
    col: np.ndarray          # C: индекс точки в M или -1
    nb: np.ndarray           # C × 3^k: индексы строк куба в ±1 шаге (включая себя) или -1
    exempt: np.ndarray       # C: «нет» на оси из OSI — край сетки не отсев (Ж-048)


def postroit_kub(coords: list, axes: dict, plan: list, base: dict) -> Kub:
    from fastbt.etap1_poisk_vorota import OSI, pkey
    shape = tuple(len(axes[a]) for a in plan)
    idx = np.indices(shape).reshape(len(plan), -1).T
    look = {pkey(c): j for j, c in enumerate(coords) if c is not None}
    pts = [{**base, **{a: axes[a][j] for a, j in zip(plan, ix)}} for ix in idx]
    col = np.array([look.get(pkey(p), -1) for p in pts])
    exempt = np.array([any(p[a] == "net" for a in OSI if a in plan) for p in pts])
    nb = np.full((len(idx), 3 ** len(plan)), -1)
    for n, offs in enumerate(itertools.product((-1, 0, 1), repeat=len(plan))):
        sh = idx + np.array(offs)
        ok = ((sh >= 0) & (sh < np.array(shape))).all(axis=1)
        nb[ok, n] = np.ravel_multi_index(tuple(sh[ok].T), shape)
    return Kub(col, nb, exempt)


def _goda(dates) -> np.ndarray:
    d = np.asarray(dates)
    return d.astype("datetime64[Y]").astype(int) + 1970 if np.issubdtype(d.dtype, np.datetime64) else d.astype(int)


def msr_stolbcy(X: np.ndarray) -> np.ndarray:
    """mSR Фавра–Галеано по столбцам X (дни × точки); то же, что metriki_e.msr (тест); nan при mVaR ≤ 0."""
    m, sd = X.mean(axis=0), X.std(axis=0, ddof=1)
    d = X - m
    m2 = (d ** 2).mean(axis=0)
    ok = m2 > 1e-24
    s = np.where(ok, (d ** 3).mean(axis=0) / np.where(ok, m2, 1.0) ** 1.5, 0.0)
    k = np.where(ok, (d ** 4).mean(axis=0) / np.where(ok, m2, 1.0) ** 2 - 3.0, 0.0)
    z = Z95 + (Z95 ** 2 - 1) * s / 6 + (Z95 ** 3 - 3 * Z95) * k / 24 - (2 * Z95 ** 3 - 5 * Z95) * s ** 2 / 36
    mvar = -(m + z * sd) * math.sqrt(252)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(mvar > 1e-12, m * 252 / mvar, np.nan)


def min_msr(M_in: np.ndarray, dates_in, rows: np.ndarray | None = None) -> np.ndarray:
    """Целевая каждой точки на половине: min mSR по годам (правило в шапке). M_in — точки × дни."""
    X = M_in if rows is None else M_in[rows]
    g = _goda(dates_in)
    per_year = [msr_stolbcy(X[:, g == y].T) for y in np.unique(g) if (g == y).sum() >= MIN_DNEY]
    return np.min(per_year, axis=0) if per_year else np.full(X.shape[0], np.nan)     # np.min: nan в любом году -> nan


def sigma126(x: np.ndarray, dates_in) -> float:
    """σ скользящего 126-дневного Шарпа (годовая шкала, ddof=1); окна — внутри непрерывных кусков половины."""
    d = np.asarray(dates_in).astype("datetime64[D]")
    cut = np.flatnonzero(np.diff(d).astype(int) > GAP_DNEY) + 1
    parts = [pd.Series(p, dtype=float) for p in np.split(np.asarray(x, float), cut)]
    r = pd.concat([(p.rolling(WIN).mean() / p.rolling(WIN).std(ddof=1) * math.sqrt(252)).dropna() for p in parts])
    return float(r.std(ddof=1)) if len(r) > 1 else float("nan")


def vybor_argmax(M_in: np.ndarray, coords, dates_in=None) -> int:
    """Справочное правило = прежнее pbo_cscv: лучшая по среднему дневному избытку, первая при равенстве."""
    return int(np.argmax(M_in.mean(axis=1)))


def vybor_centr_plato(M_in: np.ndarray, coords, dates_in, *, kub: Kub, dop: np.ndarray,
                      topk: int = TOPK, tie: float = TIE, obj_fn: Callable | None = None) -> int | None:
    """Центр плато по целевой на половине (умолчание min mSR; obj_fn(M_in, dates_in, rows) — подмена, для сверки на
    синтетике); допустимость dop[j] и «нет» на оси — как в pick_center. None — кандидатов нет."""
    used = kub.col >= 0
    o = np.full(len(kub.col), np.nan)
    o[used] = (obj_fn or min_msr)(M_in, dates_in, kub.col[used])
    fin = np.isfinite(o)
    vals = np.where(kub.nb >= 0, o[np.where(kub.nb >= 0, kub.nb, 0)], np.nan)
    n = np.isfinite(vals).sum(axis=1)
    plat = np.where(n > 0, np.nansum(vals, axis=1) / np.maximum(n, 1), np.nan)
    ok = fin & used & dop[np.where(used, kub.col, 0)] & ((n >= MIN_NEIGHBORS) | kub.exempt)
    order = [i for i in np.argsort(-np.where(np.isfinite(plat), plat, -1e9), kind="stable") if ok[i]][:topk]
    if not order:
        return None
    top = max(plat[i] for i in order if math.isfinite(plat[i]))
    near = [i for i in order if plat[i] >= top - tie]
    sg = {i: sigma126(M_in[kub.col[i]], dates_in) for i in near}
    return int(kub.col[min(near, key=lambda i: sg[i] if math.isfinite(sg[i]) else math.inf)])


def razbieniya(T: int, nblk: int):
    """Смежные блоки как в pbo_cscv (round) и все половины из nblk // 2 блоков."""
    e = [round(i * T / nblk) for i in range(nblk + 1)]
    blocks = [np.arange(e[i], e[i + 1]) for i in range(nblk)]
    for tr in itertools.combinations(range(nblk), nblk // 2):
        te = [b for b in range(nblk) if b not in tr]
        yield np.concatenate([blocks[b] for b in tr]), np.concatenate([blocks[b] for b in te])


def cscv(M: np.ndarray, coords: list, select_fn: Callable, nblk: int = 8, dates=None,
         oos_fn: Callable | None = None) -> dict:
    """pbo — доля разбиений, где выбранная in-sample точка в нижней половине рангов на out-of-sample (logit ≤ 0).
    Ранг на out-of-sample — по oos_fn(M_out, dates_out) (умолчание: среднее дневного избытка, как прежде)."""
    N, T = M.shape
    if T < nblk or N < 2:
        return dict(pbo=float("nan"), n=0, n_propusk=0, logits=[])
    dates = np.arange(T) if dates is None else np.asarray(dates)
    score = oos_fn or (lambda X, d: X.mean(axis=1))
    logits, skipped = [], 0
    for tr, te in razbieniya(T, nblk):
        best = select_fn(M[:, tr], coords, dates[tr])
        if best is None:
            skipped += 1
            continue
        a = score(M[:, te], dates[te])
        w = min(max(float(np.sum(a < a[best])) / (N - 1), 1e-6), 1 - 1e-6)
        logits.append(math.log(w / (1 - w)))
    pbo = sum(1 for x in logits if x <= 0) / len(logits) if logits else float("nan")
    return dict(pbo=pbo, n=len(logits), n_propusk=skipped, logits=logits)


def pbo_cscv_pravilo(M: np.ndarray, coords: list, select_fn: Callable, nblk: int = 8, dates=None,
                     oos_fn: Callable | None = None) -> float:
    """PBO по CSCV с выбором select_fn внутри каждого разбиения (Ж-080); oos_fn — целевая для ранга на out-of-sample."""
    return cscv(M, coords, select_fn, nblk, dates, oos_fn)["pbo"]


def pravilo_etapa(led, axes: dict, plan: list, base: dict, yakor: dict, period: str = "train") -> dict:
    """Аргументы V.pbo(M, pravilo=…) для Ledger: coords/dop в порядке столбцов V.matrix(led, period); даты — год каждого
    дня точный (excess_d по годам), внутри года порядковые сутки с 1 января (в Ledger дат нет)."""
    from fastbt import etap1_poisk_vorota as V
    ys = V.PERIOD_YEARS[period]
    coords, dop, n = [], [], {}
    for (k, per), (prm, res) in list(led.cache.items()):      # list: dopustima может досчитать якорь в Ledger
        if per != period or not set(ys) <= set(s := V.series(res)):
            continue
        n = n or {y: len(s[y]) for y in ys}
        coords.append(None if isinstance(k[0], str) else prm)
        dop.append(coords[-1] is not None and not res.get("vyrozhdena") and V.dopustima(led, prm, yakor, **V.kw(period)))
    kub = postroit_kub(coords, axes, plan, base)
    dates = np.concatenate([np.datetime64(f"{y}-01-01") + np.arange(n[y]) for y in ys]) if n else np.array([], "datetime64[D]")
    return dict(coords=coords, dates=dates, oos_fn=min_msr, select_fn=functools.partial(vybor_centr_plato, kub=kub, dop=np.array(dop, bool)))
