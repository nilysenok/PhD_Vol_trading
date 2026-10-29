"""Контроли процедуры v5 (лист П9; setka_v5 §3 «Контроли», §4 вопрос 1; otbor_statistika п. 2). Данных не читает.

Ledger здесь — минимальный контракт: r[день, тикер, точка] (дневной избыточный ряд), daty, idx[точка, ось]
(целые координаты сетки), osi (имена осей). Процедура — любая `proc(ledger, gody) -> Vyhod(dni, w)`:
w[i, тикер, точка] — вес точки в день ledger.daty[dni[i]]. Боевая — `kalibr` из fastbt/kalibr_v5.py (П7,
ещё не слита); `prostaya_kalibr` — только заглушка для синтетики: топ-K по Шарпу окна ≤ t−1.

к1 — «ничего не знающий» вход: выбор заменён безусловным средним (равный вес всех точек куба).
к2 — циклический сдвиг весов по времени (лаг ∈ [21, n−21], свой на тикер); p = (1 + #{perm ≥ точка})/(N+1).
к3 — база без новых осей: точки только с осями v4 на якорях. Остальные оси не тронуты.
Плюс: одна фиксированная точка (плато на train; совпадают годы выбора и оценки — потолок, не оценка),
второй seed Соболя (расхождение по годам), проверка причинности (префикс).
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np
from scipy.stats import qmc

from fastbt import fast_kontrol as fk
from fastbt.metriki_e import sr_dnevnoy

Ledger = namedtuple("Ledger", "r daty idx osi")
Vyhod = namedtuple("Vyhod", "dni w")
MIN_SOSEDEI = 3                                  # Ж-051: меньше трёх соседей — край, плато не берётся


def sharpe(x) -> float:
    return float(sr_dnevnoy(x) * np.sqrt(252))   # ddof = 1, √252 — одно определение (Д7)


def _god(ledger) -> np.ndarray:
    return np.asarray(ledger.daty, "datetime64[D]").astype("datetime64[Y]").astype(int) + 1970


def portfel(ledger: Ledger, v: Vyhod) -> np.ndarray:
    """Дневной ряд ансамбля: сумма весов × доходностей по точкам, среднее по тикерам."""
    return (ledger.r[v.dni] * v.w).sum(2).mean(1)


def podmnozhestvo(ledger: Ledger, mask: np.ndarray) -> Ledger:
    return Ledger(ledger.r[:, :, mask], ledger.daty, ledger.idx[mask], ledger.osi)


def prostaya_kalibr(ledger: Ledger, gody, k: int = 5, bez_vybora: bool = False) -> Vyhod:
    """Заглушка процедуры: на год t — топ-k точек по Шарпу окна ≤ t−1, по тикеру, равные веса.
    bez_vybora=True — равный вес всех точек (это и есть к1)."""
    god, (_, nt, npt) = _god(ledger), ledger.r.shape
    dni = np.flatnonzero(np.isin(god, gody))
    w = np.zeros((len(dni), nt, npt))
    for t in gody:
        pos = god[dni] == t
        if bez_vybora:
            w[pos] = 1.0 / npt
            continue
        x = ledger.r[god < t]
        sd = x.std(0, ddof=1)
        sc = np.where(sd > 1e-12, x.mean(0) / np.where(sd > 1e-12, sd, 1.0), 0.0)     # (тикер, точка)
        for ti in range(nt):
            w[np.ix_(pos, [ti], np.argsort(-sc[ti])[:k])] = 1.0 / k
    return Vyhod(dni, w)


def k1(ledger: Ledger, gody) -> float:
    """Шарп безусловного среднего по всем точкам (вход выбора — константа)."""
    return sharpe(portfel(ledger, prostaya_kalibr(ledger, gody, bez_vybora=True)))


def sdvig_vesov(w: np.ndarray, lag: np.ndarray) -> np.ndarray:
    """w'[i, тикер] = w[(i − lag[тикер]) mod n, тикер]; lag = 0 — исходные веса."""
    n = w.shape[0]
    out = np.empty_like(w)
    for ti in range(w.shape[1]):
        out[:, ti] = w[(np.arange(n) - lag[ti]) % n, ti]
    return out


def proverit_lagi(lagi: np.ndarray, n: int, min_lag: int = fk.MIN_LAG) -> None:
    if lagi.min() < min_lag or lagi.max() > n - min_lag:
        raise AssertionError(f"лаг вне [{min_lag}, {n - min_lag}]")


def k2(ledger: Ledger, v: Vyhod, n_shift: int = 199, min_lag: int = fk.MIN_LAG, lagi=None) -> dict:
    """Шарп точки, Шарпы сдвигов, p по перестановкам. lagi[b, тикер] — только для тестов (иначе draw_lags)."""
    n, nt = v.w.shape[:2]
    if lagi is None:
        lagi = fk.draw_lags(np.full(nt, n), n_shift, fk.SEED, min_lag)
        proverit_lagi(lagi, n)
    r = ledger.r[v.dni]
    perm = np.array([sharpe((r * sdvig_vesov(v.w, lag)).sum(2).mean(1)) for lag in lagi])
    pt = sharpe(portfel(ledger, v))
    return dict(sharpe=pt, perm=perm, p=fk.p_perm(pt, perm))


def k3(ledger: Ledger, imena, yakor: dict) -> Ledger:
    """База без названных осей: остаются точки, где каждая названная ось на якоре v4."""
    mask = np.ones(len(ledger.idx), bool)
    for im in imena:
        mask &= ledger.idx[:, ledger.osi.index(im)] == yakor[im]
    return podmnozhestvo(ledger, mask)


def plato(ledger: Ledger, sh: np.ndarray) -> np.ndarray:
    """Среднее Шарпа точки и её соседей (±1 по одной оси); < MIN_SOSEDEI соседей — nan (край)."""
    out = np.full(len(sh), np.nan)
    for i, c in enumerate(ledger.idx):
        d = np.abs(ledger.idx - c).sum(1)
        nb = np.flatnonzero((d == 1) & ((ledger.idx != c).sum(1) == 1))
        if len(nb) >= MIN_SOSEDEI:
            out[i] = np.mean(sh[np.r_[i, nb]])
    return out


def fiksirovannaya(ledger: Ledger, gody_vybora, gody_otsenki) -> dict:
    """Одна точка на всё: argmax плато по годам выбора, Шарп ансамбля тикеров на годах оценки."""
    god = _god(ledger)
    x = ledger.r[np.isin(god, gody_vybora)].mean(1)                  # (день, точка)
    sh = np.array([sharpe(x[:, i]) for i in range(x.shape[1])])
    pl = plato(ledger, sh)
    if np.isnan(pl).all():
        raise ValueError("нет точек с ≥ 3 соседями")
    i = int(np.nanargmax(pl))
    return dict(tochka=i, plato=float(pl[i]), sharpe=sharpe(ledger.r[np.isin(god, gody_otsenki)][:, :, i].mean(1)))


def seed2(proc, led1: Ledger, led2: Ledger, gody) -> dict:
    """Тот же расчёт на кубе со вторым seed Соболя: Шарп ансамбля по годам и расхождение."""
    if not np.array_equal(led1.daty, led2.daty):
        raise ValueError("ряды дат разные")
    out = {}
    for t in gody:
        s = [sharpe(portfel(le, proc(le, (t,)))) for le in (led1, led2)]
        out[t] = dict(sharpe1=s[0], sharpe2=s[1], rashozhdenie=abs(s[0] - s[1]))
    return out


def prichinnost(proc, ledger: Ledger, gody, seed: int = 0) -> None:
    """Веса года t не меняются, если доходности лет ≥ t испорчены (окно строго ≤ t−1)."""
    god = _god(ledger)
    for t in gody:
        bad = ledger.r.copy()
        bad[god >= t] = np.random.default_rng(seed).normal(0, 1, bad[god >= t].shape)
        a, b = proc(ledger, (t,)), proc(ledger._replace(r=bad), (t,))
        if not np.array_equal(a.w, b.w):
            raise AssertionError(f"процедура смотрит в год {t} или позже")


def sintez_ledger(seed: int, daty, levels: int = 5, osi=("a", "b", "c"), amp: float = 1.5,
                  tsentr=None, ntick: int = 3, m: int = 6, rseed: int = 1) -> Ledger:
    """Синтетический куб: точки — Соболь(seed) на сетке levels^len(osi); средний Шарп гауссовым
    плато вокруг tsentr (amp), шум общий по rseed. amp = 0 — нулевая гипотеза."""
    sob = qmc.Sobol(len(osi), scramble=True, seed=seed)
    idx = np.unique((sob.random_base2(m) * levels).astype(int), axis=0)
    c = np.full(len(osi), levels // 2) if tsentr is None else np.asarray(tsentr)
    mu = amp / np.sqrt(252) * 0.01 * np.exp(-((idx - c) ** 2).sum(1) / 2)
    full = np.random.default_rng(rseed).normal(0, 0.01, (len(daty), ntick, levels ** len(osi)))
    key = {tuple(p): j for j, p in enumerate(np.ndindex(*[levels] * len(osi)))}
    cols = [key[tuple(p)] for p in idx]                              # шум точки сетки не зависит от seed Соболя
    return Ledger(full[:, :, cols] + mu, np.asarray(daty, "datetime64[D]"), idx, tuple(osi))
