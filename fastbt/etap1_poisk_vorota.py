# -*- coding: utf-8 -*-
"""etap1_poisk_vorota.py — части драйвера поиска этапа 1 (Ж-047): кэш точек, целевая, плато, ворота Ж-044, PBO, DSR.

Контракт строки evaluate: msr_by_year {год: mSR}, excess_d {год: ряд}, sdelok_y {год: сделок}, vyrozhdena.
Периоды: "train" (2018+2019) и "ctrl2019" (только 2019). Данных не читает.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from fastbt import sweep_drivers as sd

TIE, WIN, NBLK, PBO_MAX, MIN_SDELOK = 0.05, 126, 8, 0.2, 30
OSI, NET = ("k_s", "R", "tau"), "net"


def canon(p: dict) -> dict:
    """k_s = нет -> тейка нет, R не влияет: вырожденные точки считаются один раз по эффективным параметрам."""
    q = dict(p)
    if q.get("k_s") == NET:
        q["R"] = NET
    return q


def pkey(p: dict) -> tuple:
    return tuple(sorted(canon(p).items()))


class Ledger:
    """Кэш оценок по (эффективная точка, период); N в DSR = число уникальных точек за все периоды. Только дополняется."""

    def __init__(self, evaluate):
        self.ev, self.cache = evaluate, {}

    def __call__(self, params: dict, period: str) -> dict:
        k = (pkey(params), period)
        if k not in self.cache:
            self.cache[k] = (canon(params), self.ev(canon(params), period))
        return self.cache[k][1]

    @property
    def n_unique(self) -> int:
        return len({k for k, _ in self.cache})


def obj(res: dict) -> float:
    """Целевая Ж-043: min mSR по годам окна; любой nan -> nan (точка недопустима)."""
    v = [float(x) for x in (res.get("msr_by_year") or {}).values()]
    return min(v) if v and all(math.isfinite(x) for x in v) else float("nan")


def obj_morris(res: dict) -> float:
    o = obj(res)
    return o if math.isfinite(o) else 0.0     # mSR не определён (нет сделок) = нейтрально, эффект не должен стать nan


def series(res: dict) -> dict:
    e = res.get("excess_d") or {}
    return {int(y): np.asarray(e[y], float) for y in sorted(e, key=int)}


def sigma126(res: dict) -> float:
    """σ скользящего 126-дневного Шарпа на train (2018 + 2019 подряд), годовая шкала, ddof=1."""
    s = pd.Series(np.concatenate(list(series(res).values()) or [[]]), dtype=float)
    r = (s.rolling(WIN).mean() / s.rolling(WIN).std(ddof=1) * math.sqrt(252)).dropna()
    return float(r.std(ddof=1)) if len(r) > 1 else float("nan")


def pick_best(rows: list, score: str, tol: float = TIE) -> dict | None:
    """Лучшая по score; в пределах tol от лучшей — ничья, выигрывает меньшая sigma126 (Ж-043 п. 2)."""
    ok = [r for r in rows if math.isfinite(r[score])]
    if not ok:
        return None
    top = max(r[score] for r in ok)
    return min((r for r in ok if r[score] >= top - tol), key=lambda r: r["sigma126"] if math.isfinite(r["sigma126"]) else math.inf)


def neighbors(pt: dict, names: list, axes: dict) -> list:
    """Различные (по эффективным параметрам) точки в ±1 шаге по names, кроме самой."""
    seen, out = {pkey(pt)}, []
    for offs in itertools.product((-1, 0, 1), repeat=len(names)):
        js = [axes[a].index(pt[a]) + o for a, o in zip(names, offs)]
        if all(0 <= j < len(axes[a]) for a, j in zip(names, js)):
            q = {**pt, **{a: axes[a][j] for a, j in zip(names, js)}}
            if pkey(q) not in seen:
                seen.add(pkey(q))
                out.append(q)
    return out


def n_pos(led: Ledger, pts: list) -> int:
    return sum(1 for q in pts if obj(led(q, "train")) > 0)       # nan > 0 ложно


def gate_osi(led: Ledger, pt: dict, axes: dict) -> dict:
    """Плато по k_s, R, τ отдельно (досчёт соседей в N): >= 3 положительных соседей всего и по каждой оси >= 1."""
    per = {}
    for a in OSI:
        nb = [] if (a == "R" and pt.get("k_s") == NET) else neighbors(pt, [a], axes)   # R при k_s = нет вырождена
        if nb:
            per[a] = n_pos(led, nb)
    return dict(per=per, ok=bool(per) and sum(per.values()) >= 3 and all(v >= 1 for v in per.values()))


def gates(led: Ledger, pt: dict, axes: dict, plan: list, pbo: float, s9_check) -> dict:
    sdy = led(pt, "train").get("sdelok_y") or {}
    g = dict(G1_pbo=bool(pbo < PBO_MAX), G2_sdelki=bool(sdy) and all(v >= MIN_SDELOK for v in sdy.values()),
             G3_plato=n_pos(led, neighbors(pt, plan, axes)) >= 3, G5_s9=bool(s9_check(pt)))
    o = gate_osi(led, pt, axes)
    g.update(G4_osi=o["ok"], osi_pos=str(o["per"]))
    g["ok"] = all(g[k] for k in ("G1_pbo", "G2_sdelki", "G3_plato", "G4_osi", "G5_s9"))
    return g


def matrix(led: Ledger) -> np.ndarray:
    cols = {k: np.r_[s[2018], s[2019]] for (k, per), (_, res) in led.cache.items()
            if per == "train" and {2018, 2019} <= set(s := series(res))}
    if len({len(v) for v in cols.values()}) > 1:
        raise ValueError("ряды точек разной длины: матрица T×N не строится")
    return np.column_stack(list(cols.values())) if cols else np.empty((0, 0))


def pbo(M: np.ndarray) -> float:
    return sd.pbo_cscv(M.tolist(), NBLK)["pbo"] if M.shape[1] >= 2 else float("nan")


def psr_dsr(x: np.ndarray, M: np.ndarray, n_all: int) -> dict:
    """Справочно (Ж-044): T = дней; var SR — по столбцам M, N = все посчитанные точки."""
    sr = lambda v: float(v.mean() / v.std(ddof=1)) if v.std(ddof=1) > 0 else 0.0     # noqa: E731
    if x.std(ddof=1) == 0:
        return dict(T=len(x), psr=float("nan"), dsr=float("nan"), N_dsr=n_all)
    z = (x - x.mean()) / x.std(ddof=1)
    tr = [sr(M[:, j]) for j in range(M.shape[1])]
    star = sd.expected_max_sharpe(n_all, float(np.var(tr, ddof=1)) if len(tr) > 1 else 0.0)
    a = (len(x), float((z ** 3).mean()), float((z ** 4).mean()))
    return dict(T=len(x), psr=sd.psr(sr(x), 0.0, *a), dsr=sd.psr(sr(x), star, *a), N_dsr=n_all)
