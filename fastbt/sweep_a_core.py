"""Сборка A (Ж-031, Э1-г): веса, лимиты, портфель по панели пар (тикер, стратегия). Чистый numpy, данных не читает.

Панель: pos[D,P] позиция пары (−1/0/+1), lr[D,P] лог-доходность инструмента, act[D,P] — пара есть в этот день,
strength[D,P] — сила сигнала (для равенства в день), rfc[D] — ставка ЦБ × календарные дни до следующей строки.
Веса на день d считаются только по строкам < d (окно OKNO_VESOV); текущий день и будущее в вес не входят.
Расчёт портфеля — как `s9.Engine.port` (комиссия с обеих сторон на текущий вес, плата за выход на последнем баре
пары, свободный капитал под ставку ЦБ) плюс комиссия на |Δ(w − w_равный)| удерживаемой позиции: платит только пересмотр по схеме
(обр. волатильность, паритет риска); смена 1/cnt и n_open/L не платится — у A её нет (якорь).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

OKNO_VESOV, MIN_ISTORIYA, SHRINK = 60, 10, 0.5     # дней; меньше MIN_ISTORIYA прошлых строк — равные веса
VESA = ("ravnye", "obratnaya_vol", "paritet_riska")


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    sid: np.ndarray            # [P] имя стратегии пары
    pos: np.ndarray
    lr: np.ndarray
    act: np.ndarray
    strength: np.ndarray
    rfc: np.ndarray


def erc(cov: np.ndarray, it: int = 500) -> np.ndarray:
    """Равный вклад в риск (длинный, Σw = 1): w ← w·√(цель/вклад), с нормировкой."""
    n = len(cov)
    w = np.ones(n) / n
    for _ in range(it):
        rc = w * (cov @ w)
        new = w * np.sqrt((rc.sum() / n) / np.maximum(rc, 1e-18))
        new /= new.sum()
        done, w = np.abs(new - w).max() < 1e-12, new
        if done:
            break
    return w


def slot_weights(scheme: str, hist: np.ndarray) -> np.ndarray:
    """Веса Σ = 1 для пар по прошлой истории hist[h, n] (лог-доходности инструментов, строки < d)."""
    n = hist.shape[1]
    eq = np.ones(n) / n
    if scheme == "ravnye" or len(hist) < MIN_ISTORIYA or n == 1:
        return eq
    sd = hist.std(axis=0, ddof=1)
    if not np.all(sd > 1e-12):
        return eq
    if scheme == "obratnaya_vol":
        w = 1.0 / sd
        return w / w.sum()
    cov = np.atleast_2d(np.cov(hist.T))
    cov = (1 - SHRINK) * cov + SHRINK * np.diag(np.diag(cov))
    return erc(cov)


def full_weights(pn: Panel, scheme: str) -> np.ndarray:
    """w[D,P] без лимита позиций: схема по всем парам дня (act), Σ по дню = 1; равные = 1/cnt_d, как A."""
    w = np.zeros(pn.pos.shape)
    for d in range(len(pn.dates)):
        idx = np.flatnonzero(pn.act[d])
        if len(idx):
            w[d, idx] = slot_weights(scheme, pn.lr[max(0, d - OKNO_VESOV):d][:, idx])
    return w


def gate(pn: Panel, limit, lim_strat) -> np.ndarray:
    """Допущенные позиции g[D,P]. Сделка пары, не допущенная в день входа, не открывается до конца (поздний вход
    — другая точка входа). Выход освобождает слот в тот же день; кандидаты дня — по силе сигнала, затем по номеру."""
    if limit is None and lim_strat is None:
        return pn.pos.copy()
    D, P = pn.pos.shape
    g, rej = np.zeros((D, P)), np.zeros(P, bool)
    for d in range(D):
        raw, prev = pn.pos[d], (g[d - 1] if d else np.zeros(P))
        old = pn.pos[d - 1] if d else np.zeros(P)
        keep = (prev != 0) & (raw == prev)
        rej = rej & (raw != 0) & (raw == old)
        g[d, keep] = raw[keep]
        n_open = int(keep.sum())
        per = {s: int((keep & (pn.sid == s)).sum()) for s in set(pn.sid)}
        cand = np.flatnonzero((raw != 0) & ~keep & ~rej)
        for p in sorted(cand, key=lambda q: (-pn.strength[d, q], q)):
            s = pn.sid[p]
            if (limit is None or n_open < limit) and (lim_strat is None or per[s] < lim_strat):
                g[d, p], n_open, per[s] = raw[p], n_open + 1, per[s] + 1
            else:
                rej[p] = True
    return g


def weights(pn: Panel, scheme: str, g: np.ndarray, limit, full: np.ndarray | None = None) -> np.ndarray:
    """Без лимита — `full` (по всем парам дня). С лимитом L — схема по открытым парам дня × n_open/L (равные: 1/L)."""
    if limit is None:
        return full if full is not None else full_weights(pn, scheme)
    w = np.zeros(g.shape)
    for d in range(len(pn.dates)):
        idx = np.flatnonzero(g[d] != 0)
        if len(idx):
            w[d, idx] = slot_weights(scheme, pn.lr[max(0, d - OKNO_VESOV):d][:, idx]) * len(idx) / limit
    return w


def portfolio(pn: Panel, g: np.ndarray, w: np.ndarray, comm: float, we: np.ndarray | None = None) -> pd.DataFrame:
    """[excess, exposure, turnover, sides] по дням. Σ|вес| ≤ 1: плечо запрещено (AssertionError).
    we — равный вес того же дня (без лимита 1/cnt_d по act, с лимитом 1/L); по умолчанию 1/cnt_d."""
    if we is None:
        we = pn.act / np.maximum(pn.act.sum(axis=1, keepdims=True), 1)
    dg = np.abs(np.diff(g, axis=0, prepend=0.0))
    last = np.zeros(g.shape, bool)
    for p in range(g.shape[1]):
        a = np.flatnonzero(pn.act[:, p])
        if len(a):
            last[a[-1], p] = True
    ex = np.where(last, np.abs(g), 0.0)
    gp, wp = np.vstack([np.zeros((1, g.shape[1])), g[:-1]]), np.vstack([np.zeros((1, g.shape[1])), w[:-1]])
    dev, devp = w - we, wp - np.vstack([np.zeros((1, g.shape[1])), we[:-1]])
    reb = np.where((g != 0) & (g == gp), np.abs(g) * np.abs(dev - devp), 0.0)   # платит только пересмотр по схеме, не смена 1/cnt
    # закрытая часть платит по весу дня выхода; пара, вышедшая из портфеля (w = 0, лимит), — по вчерашнему весу
    trade = np.where((gp != 0) & (g != gp), np.abs(gp), 0.0) * np.where(w > 0, w, wp) + np.where((g != 0) & (g != gp), np.abs(g), 0.0) * w
    turn = trade + ex * w
    r = (g * w * pn.lr - turn * comm - reb * comm).sum(axis=1)
    expo = (np.abs(g) * w).sum(axis=1)
    assert expo.max(initial=0.0) <= 1.0 + 1e-9, "Σ|вес| > 1: плечо не предусмотрено"
    n_act = np.maximum(pn.act.sum(axis=1), 1)
    sides = ((dg > 0) | (ex > 0)).sum(axis=1) / n_act
    return pd.DataFrame(dict(excess=r - expo * pn.rfc, exposure=expo, turnover=(turn + reb).sum(axis=1),
                             sides=sides), index=pn.dates)


def assemble(pn: Panel, p: dict, comm: float, cache: dict | None = None) -> pd.DataFrame:
    """Сборка по параметрам: vesa, limit_pozitsiy ('net'|L), limit_na_strategiyu ('net'|K)."""
    lim = None if p["limit_pozitsiy"] == "net" else int(p["limit_pozitsiy"])
    ks = None if p["limit_na_strategiyu"] == "net" else int(p["limit_na_strategiyu"])
    g = gate(pn, lim, ks)
    cache = {} if cache is None else cache
    if lim is None and p["vesa"] not in cache:
        cache[p["vesa"]] = full_weights(pn, p["vesa"])       # без лимита веса от сигналов не зависят: один раз на схему
    full = cache.get(p["vesa"]) if lim is None else None
    we = None if lim is None else (g != 0) / lim
    return portfolio(pn, g, weights(pn, p["vesa"], g, lim, full), comm, we)
