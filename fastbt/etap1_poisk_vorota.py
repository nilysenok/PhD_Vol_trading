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

TIE, WIN, NBLK, PBO_MAX, MIN_SDELOK, PLATO_NET = 0.05, 126, 8, 0.2, 30, 0.1
OSI, NET = ("k_s", "R", "tau"), "net"
OUTER = {"k_s": (0.75, 0.5), "R": (0.75, 0.5)}      # Ж-049: шаг наружу от края сетки; у τ = 1 день его нет


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


def net_plato(led: Ledger, pt: dict, a: str, axes: dict) -> bool:
    """Ж-048 п. 1: при «нет» на оси плато одностороннее — два ближайших конечных значения оси (k_s 4,3; R 4,3; τ 20,10)
    не хуже точки больше чем на PLATO_NET по целевой. Недосчитанные соседи досчитываются через Ledger (идут в N)."""
    fin = [v for v in axes[a] if v != NET][-2:]
    o = obj(led(pt, "train"))
    return bool(fin) and all(obj(led({**pt, a: v}, "train")) >= o - PLATO_NET for v in fin)     # nan -> ложь


def kray_plato(led: Ledger, pt: dict, a: str) -> bool | None:
    """Ж-049: крайнее конечное значение (k_s, R, τ = 1). True — наружу шагнуть можно (OUTER): внешние точки — обычные соседи
    оси в общем правиле «>= 3 в сумме, >= 1 на оси», отдельного условия нет; None — нельзя (τ = 1 день), плато не проходит."""
    return None if a not in OUTER else True


def gate_osi(led: Ledger, pt: dict, axes: dict) -> dict:
    """Плато по k_s, R, τ: конечные значения — >= 3 положительных соседей в сумме и >= 1 на оси (Ж-048; крайнее значение —
    метка kray, соседи включают шаг наружу, Ж-049); ось с «нет» — одностороннее плато net_plato, в счёт >= 3 не входит."""
    per, net, kray = {}, {}, {}
    for a in OSI:
        if a == "R" and pt.get("k_s") == NET:
            continue                                                                  # R при k_s = нет вырождена
        if pt.get(a) == NET:
            net[a] = net_plato(led, pt, a, axes)
        elif nb := neighbors(pt, [a], axes):
            edge = pt[a] == axes[a][0]
            if edge:
                kray[a] = kray_plato(led, pt, a)
                nb += [{**pt, a: v} for v in OUTER.get(a, ())]
            per[a] = n_pos(led, nb)
    ok = bool(per or net) and all(net.values()) and all(v >= 1 for v in per.values()) and (not per or sum(per.values()) >= 3)
    ok = ok and all(v is True for v in kray.values())
    out = {**per, **{f"{a}_net": v for a, v in net.items()}, **{f"{a}_kray": (v if v is not None else "край сетки τ = 1") for a, v in kray.items()}}
    return dict(per=out, ok=ok, kray=" ".join(kray) or "-")


def gates(led: Ledger, pt: dict, axes: dict, plan: list, pbo: float, s9_check) -> dict:
    sdy = led(pt, "train").get("sdelok_y") or {}
    g = dict(G1_pbo=bool(pbo < PBO_MAX), G2_sdelki=bool(sdy) and all(v >= MIN_SDELOK for v in sdy.values()),
             G3_plato=n_pos(led, neighbors(pt, plan, axes)) >= 3, G5_s9=bool(s9_check(pt)))
    o = gate_osi(led, pt, axes)
    g.update(G4_osi=o["ok"], osi_pos=str(o["per"]), kray_setki=o["kray"])
    g["ok"] = all(g[k] for k in ("G1_pbo", "G2_sdelki", "G3_plato", "G4_osi", "G5_s9"))
    return g


def storony(res: dict) -> dict:
    """Сторон в год; если драйвер оценки не отдал storony_y — 2 × сделок (так считает etap1_eval)."""
    s = res.get("storony_y")
    return dict(s) if s else {y: 2 * v for y, v in (res.get("sdelok_y") or {}).items()}


def dopustima(led: Ledger, pt: dict, yakor: dict) -> bool:
    """Толкование 27.10 19:35: сторон/год в каждом году >= 0,5 × сторон якоря на 2018; плюс Ж-044: сделок >= MIN_SDELOK в год."""
    base = storony(led(yakor, "train")).get(2018)
    res = led(pt, "train")
    st, sdy = storony(res), res.get("sdelok_y") or {}
    if base is None or not math.isfinite(base) or not st or not sdy:
        return False
    return all(v >= 0.5 * base for v in st.values()) and all(v >= MIN_SDELOK for v in sdy.values())


def po_godam(res: dict) -> dict:
    """Колонки метрик по годам для факторного файла и итога: sharpe, msr, storon, sdelok."""
    nan, st, sdy = float("nan"), storony(res), res.get("sdelok_y") or {}
    cols = {"sharpe": res.get("by_year") or {}, "msr": res.get("msr_by_year") or {}, "storon": st, "sdelok": sdy}
    return {f"{n}_{y}": d.get(y, nan) for n, d in cols.items() for y in (2018, 2019)}


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
