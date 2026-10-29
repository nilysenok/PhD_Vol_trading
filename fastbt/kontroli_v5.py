"""Контроли процедуры v5 (лист П9; setka_v5 §3 «Контроли», §4 вопрос 1; otbor_statistika п. 2). Данных не читает.

Работает поверх `kalibr_v5.Pul` (ex[P, тикеры, T] — дневной избыточный ряд точек стратегии) и
`kalibr_v5.god` (отбор года t по окну [начало; 31.12(t−1)]). Ансамбль = равный вес выбранных членов
(`chleny`), как их `ryad`; позиций в ledger v2.0 нет, `vote` не нужен. ↺ класс B: при позициях контроли
считаются по ряду, не по `vote` (откат: считать через `K.ansambl`).

к1 — «ничего не знающий» вход: выбор заменён безусловным средним (равный вес всех точек пула).
к2 — циклический сдвиг весов выбора по времени (лаг ∈ [21, n−21], свой на тикер, seed и границы как у
     s9-к2: fast_kontrol); p = (1 + #{perm ≥ точка})/(N+1). Сдвиг ломает связь «когда выбрано» — «когда
     доходность»; стационарный выбор (один и тот же холм каждый год) к2 не опровергает: это честный ответ.
к3 — база без новых осей: оставляются точки, где названные оси на якоре v4; ось в osi пула сворачивается
     в якорь (соседей по ней нет), остальные оси не тронуты.
+ одна фиксированная точка (плато на годах выбора; годы выбора = оценки — потолок, не оценка),
второй seed Соболя (расхождение Шарпа по годам), причинность (веса года t не зависят от лет ≥ t).
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np
from scipy.stats import qmc

from fastbt import fast_kontrol as fk
from fastbt import kalibr_v5 as K
from fastbt import ledger_v2 as lm
from fastbt.data import year_bounds
from fastbt.metriki_e import sr_dnevnoy

Vyhod = namedtuple("Vyhod", "dni w")             # w[i, тикер, точка] — вес в день daty[dni[i]]


def parametry(period: str = "train4") -> tuple:
    """(годы walk-forward, начало окна отбора) периода из `kalibr_v5.PERIODY`: kalibr(pul, *parametry("wf"))."""
    nachalo, gody = K.PERIODY[period]
    return gody, nachalo


GODY, NACHALO = parametry()


def sharpe(x) -> float:
    return float(sr_dnevnoy(x) * np.sqrt(252))   # ddof = 1, √252 — одно определение (Д7)


def _god(pul) -> np.ndarray:
    return pul.daty.astype("datetime64[Y]").astype(int) + 1970


def portfel(pul, v: Vyhod) -> np.ndarray:
    """Дневной ряд ансамбля: Σ вес·доходность по точкам, среднее по тикерам."""
    return (pul.ex.transpose(2, 1, 0)[v.dni] * v.w).sum(2).mean(1)


def kalibr(pul, gody=GODY, nachalo: str = NACHALO) -> Vyhod:
    """Боевая процедура П7 → веса: на год t каждому тикеру — равный вес `chleny` из K.god."""
    dni, w, nt = [], [], pul.ex.shape[1]
    for t in gody:
        r = K.god(pul, t, nachalo)
        s, e = year_bounds(pul.daty, t)
        wt = np.zeros((e - s, nt, pul.ex.shape[0]))
        for k, x in r["tikery"].items():
            if x:
                wt[:, k, [pul.idx[c] for c in x["chleny"]]] = 1.0 / len(x["chleny"])
        dni.append(np.arange(s, e))
        w.append(wt)
    return Vyhod(np.concatenate(dni), np.concatenate(w))


def k1(pul, gody=GODY) -> float:
    """Шарп безусловного среднего по всем точкам пула на годах gody."""
    return sharpe(pul.ex[:, :, np.isin(_god(pul), gody)].mean((0, 1)))


def sdvig_vesov(w: np.ndarray, lag: np.ndarray) -> np.ndarray:
    """w'[i, тикер] = w[(i − lag[тикер]) mod n, тикер]; lag = 0 — исходные веса."""
    n, out = w.shape[0], np.empty_like(w)
    for ti in range(w.shape[1]):
        out[:, ti] = w[(np.arange(n) - lag[ti]) % n, ti]
    return out


def proverit_lagi(lagi: np.ndarray, n: int, min_lag: int = fk.MIN_LAG) -> None:
    if lagi.min() < min_lag or lagi.max() > n - min_lag:
        raise AssertionError(f"лаг вне [{min_lag}, {n - min_lag}]")


def k2(pul, v: Vyhod, n_shift: int = 199, min_lag: int = fk.MIN_LAG, lagi=None) -> dict:
    """Шарп точки, Шарпы сдвигов, p. lagi[b, тикер] задают только тесты (иначе fk.draw_lags)."""
    n, nt = v.w.shape[:2]
    if lagi is None:
        lagi = fk.draw_lags(np.full(nt, n), n_shift, fk.SEED, min_lag)
        proverit_lagi(lagi, n)
    r = pul.ex.transpose(2, 1, 0)[v.dni]
    perm = np.array([sharpe((r * sdvig_vesov(v.w, lag)).sum(2).mean(1)) for lag in lagi])
    pt = sharpe(portfel(pul, v))
    return dict(sharpe=pt, perm=perm, p=fk.p_perm(pt, perm))


def podpul(pul, mask: np.ndarray, osi: dict | None = None):
    poz = None if pul.poz is None else pul.poz[mask]
    return K.Pul(pul.sid, pul.daty, pul.osi if osi is None else osi, [p for p, m in zip(pul.tochki, mask) if m],
                 pul.ex[mask], poz, pul.komissiya)


def k3(pul, imena, yakor: dict):
    """База без новых осей: точки с названными осями на якоре; ось в osi свёрнута в якорь."""
    mask = np.ones(len(pul.tochki), bool)
    for a in imena:
        mask &= pul.kol[a] == lm.tok(yakor[a])
    return podpul(pul, mask, {a: ([lm.tok(yakor[a])] if a in imena else vs) for a, vs in pul.osi.items()})


def plato(pul, sh: np.ndarray) -> np.ndarray:
    """Среднее Шарпа точки и её соседей (±1 позиция по одной оси) в пуле; < MIN_NB соседей — nan (край)."""
    ax = list(pul.osi)
    c = np.array([[pul.osi[a].index(pt[a]) for a in ax] for pt in pul.tochki])
    out = np.full(len(sh), np.nan)
    for i in range(len(c)):
        d = np.abs(c - c[i])
        nb = np.flatnonzero((d.sum(1) == 1) & ((d > 0).sum(1) == 1))
        if len(nb) >= K.MIN_NB:
            out[i] = np.mean(sh[np.r_[i, nb]])
    return out


def fiksirovannaya(pul, gody_vybora, gody_otsenki) -> dict:
    """Одна точка на всё: argmax плато по годам выбора; Шарп ансамбля тикеров на годах оценки."""
    god = _god(pul)
    sh = K.sharp_okna(pul.ex[:, :, np.isin(god, gody_vybora)].mean(1))
    pl = plato(pul, sh)
    if np.isnan(pl).all():
        raise ValueError("нет точек с достаточным числом соседей")
    i = int(np.nanargmax(pl))
    return dict(tochka=i, plato=float(pl[i]), sharpe=sharpe(pul.ex[i][:, np.isin(god, gody_otsenki)].mean(0)))


def seed2(proc, pul1, pul2, gody=GODY) -> dict:
    """Тот же расчёт на кубе со вторым seed Соболя: Шарп ансамбля по годам и расхождение."""
    if not np.array_equal(pul1.daty, pul2.daty):
        raise ValueError("ряды дат разные")
    out = {}
    for t in gody:
        s = [sharpe(portfel(p, proc(p, (t,)))) for p in (pul1, pul2)]
        out[t] = dict(sharpe1=s[0], sharpe2=s[1], rashozhdenie=abs(s[0] - s[1]))
    return out


def prichinnost(proc, pul, gody=GODY, seed: int = 0) -> None:
    """Веса года t не меняются, если доходности лет ≥ t испорчены (окно строго ≤ t−1)."""
    god = _god(pul)
    for t in gody:
        ex = pul.ex.copy()
        m = god >= t
        ex[:, :, m] = np.random.default_rng(seed).normal(0, 0.01, ex[:, :, m].shape)
        a, b = proc(pul, (t,)), proc(podpul_ex(pul, ex), (t,))
        if not np.array_equal(a.w, b.w):
            raise AssertionError(f"процедура смотрит в год {t} или позже")


def podpul_ex(pul, ex: np.ndarray):
    return K.Pul(pul.sid, pul.daty, pul.osi, pul.tochki, ex, pul.poz, pul.komissiya, pul.dosschitat)


def sintez_pul(seed: int, daty, osi=None, amp: float = 1.5, m: int = 7, ntick: int = 2, rseed: int = 1):
    """Синтетический пул: точки — Соболь(seed) на сетке osi; Шарп — гауссов холм amp в середине сетки (amp = 0 — H0).
    Шум точки сетки не зависит от seed; соседей вне выборки досчитывает `dosschitat` (как боевой)."""
    osi = osi or {"a": list(range(5)), "b": list(range(5)), "c": list(range(4))}
    ax, sz = list(osi), [len(v) for v in osi.values()]
    full = np.random.default_rng(rseed).normal(0, 0.01, (int(np.prod(sz)), ntick, len(daty)))
    mid = np.array([(n - 1) / 2 for n in sz])

    def rows(cells):
        mu = amp / np.sqrt(252) * 0.01 * np.exp(-((np.array(cells) - mid) ** 2).sum(1) / 2)
        return full[[np.ravel_multi_index(c, sz) for c in cells]] + mu[:, None, None]
    u = qmc.Sobol(len(ax), scramble=True, seed=seed).random_base2(m)
    cells = np.unique((u * sz).astype(int), axis=0)
    pts = lambda cs: [{a: osi[a][j] for a, j in zip(ax, c)} for c in cs]            # noqa: E731
    dos = lambda pt: (rows([[osi[a].index(pt[a]) for a in ax]])[0], None)           # noqa: E731
    return K.Pul("S1", daty, osi, pts(cells), rows(cells), None, 0.0, dos)
