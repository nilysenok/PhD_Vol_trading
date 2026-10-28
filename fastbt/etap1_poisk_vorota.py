# -*- coding: utf-8 -*-
"""etap1_poisk_vorota.py — части драйвера поиска этапа 1 (Ж-047): кэш точек, целевая, плато, ворота Ж-044, PBO, DSR.

Контракт строки evaluate: msr_by_year {год: mSR}, excess_d {год: ряд}, sdelok_y {год: сделок}, vyrozhdena. Периоды: train, ctrl2019, train4 (В1). Данных не читает.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from fastbt import sweep_drivers as sd
from fastbt.etap1_data import psr_dsr  # noqa: F401  (перенесено в etap1_data: лимит 200 строк; V.psr_dsr остаётся)

TIE, WIN, NBLK, PBO_MAX, MIN_SDELOK, PLATO_NET = 0.05, 126, 8, 0.2, 30, 0.1
OSI, NET = ("k_s", "R", "tau"), "net"
PERIOD_YEARS = {"train": (2018, 2019), "ctrl2019": (2019,), "tune2018": (2018,), "train4": (2016, 2017, 2018, 2019)}
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
    """Кэш оценок по (эффективная точка, период); N в DSR = число уникальных точек за периоды. Только дополняется. Д5: комиссия (evaluate.comm) в ключе; нет — 0,04 %."""

    def __init__(self, evaluate, comm=None):
        self.ev, self.cache, self.comm = evaluate, {}, (getattr(evaluate, "comm", 0.0004) if comm is None else comm)

    def __call__(self, params: dict, period: str) -> dict:
        k = (pkey(params) + (() if self.comm == 0.0004 else (("komissiya", self.comm),)), period)
        if k not in self.cache:
            self.cache[k] = (canon(params), self.ev(canon(params), period))
        return self.cache[k][1]

    @property
    def n_unique(self) -> int:
        return len({k for k, _ in self.cache})


def obj(res: dict, period: str | None = None) -> float:
    """Ж-043: min mSR по имеющимся годам (None/«train» — как раньше, ctrl2019 — один год); train4 — по всем 4, нехватка -> nan."""
    m = {int(y): float(x) for y, x in (res.get("msr_by_year") or {}).items()}
    if period not in (None, "train"):
        m = {y: m.get(y, float("nan")) for y in PERIOD_YEARS[period]}
    v = list(m.values())
    return min(v) if v and all(math.isfinite(x) for x in v) else float("nan")


def obj_morris(res: dict, period: str | None = None) -> float:
    o = obj(res, period)
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


def kw(period: str) -> dict:     # period — только вне умолчания: путь «train» вызывает n_pos/one_sided ровно как раньше
    return {} if period == "train" else {"period": period}


def n_pos(led: Ledger, pts: list, period: str = "train") -> int:
    return sum(1 for q in pts if obj(led(q, period), period) > 0)       # nan > 0 ложно


def one_sided(led: Ledger, pt: dict, a: str, vals: list, period: str = "train") -> bool:
    """Одностороннее плато: точки pt[a] -> vals не хуже точки больше чем на PLATO_NET по целевой. Недосчитанные
    соседи досчитываются через Ledger (идут в N)."""
    o = obj(led(pt, period), period)
    return bool(vals) and all(obj(led({**pt, a: v}, period), period) >= o - PLATO_NET for v in vals)     # nan -> ложь


def net_plato(led: Ledger, pt: dict, a: str, axes: dict, period: str = "train") -> bool:
    """Ж-048 п. 1: при «нет» на оси — два ближайших конечных значения оси (k_s 4,3; R 4,3; τ 20,10)."""
    return one_sided(led, pt, a, [v for v in axes[a] if v != NET][-2:], **kw(period))


def kray_plato(led: Ledger, pt: dict, a: str) -> bool:
    """Ж-049: k_s = 1 и R = 1 — край сетки с шагом наружу (OUTER): внешние точки — обычные соседи оси в общем правиле
    «>= 3 в сумме, >= 1 на оси». Ж-050: τ = 1 — физический край, одностороннее плато τ = 2, 3 (в gate_osi), не отказ."""
    return True


def gate_osi(led: Ledger, pt: dict, axes: dict, period: str = "train") -> dict:
    """Плато по k_s, R, τ: конечные значения — >= 3 положительных соседей в сумме и >= 1 на оси (Ж-048; крайнее значение —
    метка kray, соседи включают шаг наружу, Ж-049); ось с «нет» — одностороннее плато net_plato, в счёт >= 3 не входит."""
    per, net, kray, fiz = {}, {}, {}, {}
    for a in OSI:
        if a == "R" and pt.get("k_s") == NET:
            continue                                                                  # R при k_s = нет вырождена
        if pt.get(a) == NET:
            net[a] = net_plato(led, pt, a, axes, **kw(period))
        elif nb := neighbors(pt, [a], axes):
            edge = pt[a] == axes[a][0]
            if edge:
                kray[a] = kray_plato(led, pt, a)
            if edge and a == "tau":                                                   # Ж-050: τ = 1 — τ = 2, 3 не хуже на 0,1
                fiz[a] = one_sided(led, pt, a, axes[a][1:3], **kw(period))
                continue
            per[a] = n_pos(led, nb + ([{**pt, a: v} for v in OUTER.get(a, ())] if edge else []), **kw(period))
    ok = bool(per or net or fiz) and all(net.values()) and all(fiz.values()) and all(v >= 1 for v in per.values())
    ok = ok and (not per or sum(per.values()) >= 3) and all(kray.values())
    out = {**per, **{f"{a}_net": v for a, v in net.items()}, **{f"{a}_fiz": v for a, v in fiz.items()},
           **{f"{a}_kray": v for a, v in kray.items() if a in OUTER}}
    return dict(per=out, ok=ok, kray=" ".join(kray) or "-")


def gates(led: Ledger, pt: dict, axes: dict, plan: list, pbo: float, s9_check, period: str = "train") -> dict:
    sdy = led(pt, period).get("sdelok_y") or {}
    g = dict(G1_pbo=bool(pbo < PBO_MAX), G2_sdelki=sdelok_ok(sdy, period),
             G3_plato=n_pos(led, neighbors(pt, plan, axes), **kw(period)) >= 3, G5_s9=bool(s9_check(pt)))
    o = gate_osi(led, pt, axes, period)
    g.update(G4_osi=o["ok"], osi_pos=str(o["per"]), kray_setki=o["kray"])
    g["ok"] = all(g[k] for k in ("G1_pbo", "G2_sdelki", "G3_plato", "G4_osi", "G5_s9"))
    return g


def sdelok_ok(sdy: dict, period: str = "train") -> bool:
    """Ж-044 G2: сделок >= MIN_SDELOK в каждом году (train — в имеющихся, как раньше; train4 — нет года = не пройдено)."""
    full = period == "train" or set(PERIOD_YEARS[period]) <= {int(y) for y in sdy}
    return bool(sdy) and full and all(v >= MIN_SDELOK for v in sdy.values())


def storony(res: dict) -> dict:
    """Сторон в год; если драйвер оценки не отдал storony_y — 2 × сделок (так считает etap1_eval)."""
    s = res.get("storony_y")
    return dict(s) if s else {y: 2 * v for y, v in (res.get("sdelok_y") or {}).items()}


def dopustima(led: Ledger, pt: dict, yakor: dict, period: str = "train") -> bool:
    """Толкование 27.10 19:35: сторон/год в каждом году >= 0,5 × сторон якоря на 2018; плюс Ж-044: сделок >= MIN_SDELOK в год."""
    base = storony(led(yakor, "train")).get(2018)     # якорь s9 — только 2018–2019: на 2016–2017 эталона s9 нет
    res = led(pt, period)
    st, sdy = storony(res), res.get("sdelok_y") or {}
    if base is None or not math.isfinite(base) or not st or not sdy:
        return False
    return all(v >= 0.5 * base for v in st.values()) and sdelok_ok(sdy, period)


def po_godam(res: dict, period: str = "train") -> dict:
    """Колонки метрик по годам для факторного файла и итога: sharpe, msr, storon, sdelok."""
    nan, st, sdy = float("nan"), storony(res), res.get("sdelok_y") or {}
    cols = {"sharpe": res.get("by_year") or {}, "msr": res.get("msr_by_year") or {}, "storon": st, "sdelok": sdy}
    return {f"{n}_{y}": d.get(y, nan) for n, d in cols.items() for y in PERIOD_YEARS[period]}


def matrix(led: Ledger, period: str = "train") -> np.ndarray:
    """T×N по годам периода подряд (train — 2018+2019, train4 — 2016…2019)."""
    ys = PERIOD_YEARS[period]
    cols = {k: np.concatenate([s[y] for y in ys]) for (k, per), (_, res) in led.cache.items()
            if per == period and set(ys) <= set(s := series(res))}
    if len({len(v) for v in cols.values()}) > 1:
        raise ValueError("ряды точек разной длины: матрица T×N не строится")
    return np.column_stack(list(cols.values())) if cols else np.empty((0, 0))


def pbo(M: np.ndarray, nblk: int = NBLK) -> float:
    return sd.pbo_cscv(M.tolist(), nblk)["pbo"] if M.shape[1] >= 2 else float("nan")
