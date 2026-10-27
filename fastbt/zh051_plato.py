"""Плато и ворота итераций 3, 4, 10 Ж-051 — как в итерации 1 (`etap1_poisk.pick_center`, `etap1_poisk_vorota.gates`), без
отступлений (Ж-051 п. 3, ворота 22:20 не меняются): плато = среднее целевой по соседям ±1 (`sd.plateau_score`), край (< 3 точек
с самой) не выбирается; G3 — не меньше 3 соседей с min mSR > 0; G4 по k_s, R, τ — `gate_osi` (Ж-048…050). Нет соседей — пометка
`net_sosedey`, плато не пройдено. Выключенные значения осей итерации = P1; псевдонимы P1 не плодят N (`PlusLedger`).
"""
from __future__ import annotations

import itertools
import math

from fastbt import etap1_poisk as P, etap1_poisk_vorota as V, sweep_drivers as sd
from fastbt.etap1_eval import NET

PLUS_KEYS = ("top_k", "sektor_trend", "shirina", "ma_n", "porog_shiriny")
GATES = ("G1_pbo", "G2_sdelki", "G3_plato", "G4_osi", "G5_s9")


def norm(ext: dict):
    """Оси итерации → каноническая точка; всё выключено — None (= P1); shirina = net → порог не влияет."""
    e = dict(ext)
    if e.get("sektor_trend", NET) == NET and e.get("shirina", NET) == NET and e.get("top_k", NET) == NET:
        return None
    if e.get("shirina", NET) == NET and "porog_shiriny" in e:
        e["porog_shiriny"] = 0.5
    return e


def canon_plus(p: dict) -> dict:
    n = norm({k: p[k] for k in PLUS_KEYS if k in p})
    return {**{k: v for k, v in p.items() if k not in PLUS_KEYS}, **(n or {})}


class PlusLedger(V.Ledger):
    """Ledger, в котором точки с выключенными осями итерации совпадают с P1 (один ключ, один раз в N)."""

    def __call__(self, params: dict, period: str) -> dict:
        return super().__call__(canon_plus(params), period)


def uniq_nbrs(p1: dict, e: dict, grid: dict) -> list:
    """Соседи ±1 по осям итерации: различные полные точки (значение «net» — это P1), без самой точки."""
    seen, out = {V.pkey(canon_plus({**p1, **e}))}, []
    for q in V.neighbors(e, list(grid), grid):
        c = canon_plus({**p1, **q})
        if V.pkey(c) not in seen:
            seen.add(V.pkey(c))
            out.append(c)
    return out


def gates_plus(led, p1: dict, e: dict, grid: dict, axes: dict, pb: float, s9_ok: bool) -> dict:
    pt = {**p1, **e}
    sdy = led(pt, "train").get("sdelok_y") or {}
    nb, o = uniq_nbrs(p1, e, grid), V.gate_osi(led, pt, axes)
    g = dict(G1_pbo=bool(pb < V.PBO_MAX), G2_sdelki=bool(sdy) and all(v >= V.MIN_SDELOK for v in sdy.values()),
             G3_plato=V.n_pos(led, nb) >= 3, G4_osi=o["ok"], G5_s9=bool(s9_ok), osi_pos=str(o["per"]), n_sosedey=len(nb))
    return {**g, "net_sosedey": len(nb) < 3, "ok": all(g[k] for k in GATES)}      # < 3 соседей: плато не пройти


def vybor(led, p1: dict, grid: dict, axes: dict, yakor: dict, s9_ok: bool):
    """(лучшая кандидатка | None, все кандидаты). Кандидатки — канонические точки сетки итерации, допустимые по активности."""
    keys, combos = list(grid), [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    rows = [{**e, "objective": V.obj(led({**p1, **e}, "train"))} for e in combos]
    pb, cands = V.pbo(V.matrix(led)), []
    for e in combos:
        if norm(e) != e:
            continue                                           # псевдоним P1 или дубль порога при shirina = net
        r, o = led({**p1, **e}, "train"), V.obj(led({**p1, **e}, "train"))
        ps, n = sd.plateau_score(rows, keys, grid, e)
        cands.append(dict(e=e, objective=o, plateau=ps, n=n, sigma126=V.sigma126(r), edge=n < 3, net_sosedey=False,
                          adm=math.isfinite(o) and V.dopustima(led, {**p1, **e}, yakor) and not r.get("vyrozhdena")))
    ok = [c for c in cands if c["adm"] and not c["edge"] and math.isfinite(c["plateau"])]
    top = sorted(ok, key=lambda c: -c["plateau"])[:P.TOPK]
    for c in top:
        c["gates"] = gates_plus(led, p1, c["e"], grid, axes, pb, s9_ok)
        c["net_sosedey"] = c["gates"]["net_sosedey"]
    best = V.pick_best([c for c in top if c["gates"]["ok"]], "plateau") or V.pick_best(top, "plateau")
    if best is None:                                            # все допустимые — край (< 3 точек с самой): плато не пройдено
        best = V.pick_best([c for c in cands if c["adm"] and math.isfinite(c["plateau"])], "plateau")
        if best is not None:
            g = gates_plus(led, p1, best["e"], grid, axes, pb, s9_ok)
            best.update(net_sosedey=True, gates={**g, "G3_plato": False, "ok": False})
    return best, cands
