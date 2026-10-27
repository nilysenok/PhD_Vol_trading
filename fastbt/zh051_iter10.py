"""Итерация 10 Ж-051: сочетание прошедших элементов 2 (ансамбль), 3 (топ-k), 4 (сектор/ширина) — план «вкл/выкл».

Вход: ledger 2, 3, 4 (sha256 сверяется), таблицы `it{2,3,4}/{S}_itog.csv`. Точек не больше 2³ − 1 = 7 новых (одиночный
ансамбль уже в ledger 2). Общий PBO и N — по объединённой матрице 1 + 2 + 3 + 4 + 10. Центр плато по факторному плану
не строится (3 оси «вкл/выкл» — решётка без соседей по шагу) — выбор по целевой с ничьей по σ126, G3 не применяется.
"""
from __future__ import annotations

import datetime as dt
import itertools
import math

import pandas as pd

from fastbt import etap1, etap1_ansambl as A, etap1_poisk as P, etap1_poisk_vorota as V, zh051_kriteriy as K
from fastbt.etap1_eval import YAKOR
from fastbt.etap1_plus_eval import PlusEvaluator
from fastbt.zh051_iter import OUT, PLUS_KEYS

NUM = {"top_k": int, "ma_n": int, "porog_shiriny": float}


def last(it: int, sid: str, name: str = "itog") -> dict:
    return pd.read_csv(OUT / f"it{it}" / f"{sid}_{name}.csv", comment="#", dtype=str).iloc[-1].to_dict()


def _num(v: str):
    if v in ("True", "False"):
        return v == "True"
    try:
        return float(v)
    except ValueError:
        return v


def ext_of(r: dict) -> dict:
    """Оси Pi из строки itog итерации 3/4 (`pi_*`), пустые пропущены."""
    return {a: NUM.get(a, str)(r[f"pi_{a}"]) for a in PLUS_KEYS if isinstance(r.get(f"pi_{a}"), str) and r[f"pi_{a}"] != ""}


def elementy(sid: str) -> dict:
    """Прошедшие элементы: имя → оси (ансамбль — строка it2)."""
    r2 = {k: _num(v) for k, v in last(2, sid).items()}
    out = {"ans": r2} if K.elem_it2(r2)["proshla"] else {}
    for name, it in (("top_k", 3), ("sektor", 4)):
        r = last(it, sid)
        if r.get("proshla") == "True":
            out[name] = ext_of(r)
    return out


def kombinacii(E: list) -> list:
    """Непустые подмножества E, кроме одного ансамбля (он уже в ledger 2); порядок — по числу включённых."""
    sub = [c for n in range(1, len(E) + 1) for c in itertools.combinations(E, n)]
    return [c for c in sub if c != ("ans",)]


def run_10(sid: str) -> dict:
    S, run_id = P.load_setka(sid), dt.datetime.now().isoformat(timespec="seconds")
    evr = PlusEvaluator(sid, etap1.build_inputs(sid), etap1.base_point(sid), 0.0004)
    ev = lambda p, per: evr.evaluate({**YAKOR, **p}, per)                    # noqa: E731
    led = K.load_ledger(sid, "ledger_it2.pkl", ev)
    for n in (3, 4):
        K.merge(led, K.load_ledger(sid, f"ledger_it{n}.pkl", ev))
    if K.sverka_ledger(led, ev):
        raise SystemExit(f"{sid}: СТОП, строки ledger не воспроизведены текущим кодом")
    row = pd.read_csv(OUT.parent / "zh047" / f"{sid}_itog_v2.csv", comment="#").iloc[-1]
    p1 = {k: P._coerce(str(row[f"p_{k}"]), S["axes"][k]) for k in S["axes"]}
    pb0, n0, E = V.pbo(V.matrix(led)), led.n_unique, elementy(sid)
    r1, base = led(p1, "train"), V.storony(led(S["yakor"], "train")).get(2018)
    cfg, ens = {}, E.get("ans")
    for c in kombinacii(list(E)):
        ext = {k: v for n in c if n != "ans" for k, v in E[n].items()}
        if "ans" in c:
            osi = pd.read_csv(OUT / "it2" / f"{sid}_ansambl.csv", comment="#", dtype=str)
            osi = osi[(osi.nabor == ens["nabor"]) & (osi.rezhim_ans == ens["rezhim_ans"])].iloc[-1]["osi"].split()
            sig = A.signal_points(A.tochki(p1, osi, S["axes"]), evr.sk.sig_names)
            res = evr.evaluate_ens({**p1, **ext}, sig, ens["rezhim_ans"])
            K.add_cfg(led, ("kombo", c), {"kombo": c, **ext}, res)
        else:
            res = led({**p1, **ext}, "train")
        cfg[c] = (res, ext)
    itog = dict(strategiya=sid, zapusk=run_id, iteraciya=10, N_do=n0, pbo_do=pb0, elementy=" ".join(E) or "нечего сочетать")
    cand = [dict(c=c, objective=V.obj(r), sigma126=V.sigma126(r)) for c, (r, _) in cfg.items()
            if all(v >= 0.5 * base for v in V.storony(r).values()) and all(v >= V.MIN_SDELOK for v in r["sdelok_y"].values())]
    best = V.pick_best(cand, "objective")
    M = V.matrix(led)
    itog.update(pbo_obshchaya=V.pbo(M), N_obshchee=led.n_unique, N_pbo=M.shape[1], G1_pbo=bool(V.pbo(M) < V.PBO_MAX))
    if best is None:
        itog.update(proshla=False, G4_osi=False, G5_s9=False, verdikt="нечего сочетать" if not E else "не прошла: нет допустимой")
    else:
        rb, ext = cfg[best["c"]]
        g4 = V.gate_osi(led, {**p1, **ext}, S["axes"])
        s9 = max(etap1.anchor_s9_diff(sid, evr).values()) <= 1e-12
        d = best["objective"] - V.obj(r1)
        ok = itog["G1_pbo"] and g4["ok"] and s9 and d >= K.DELTA - 1e-12 and math.isfinite(d)
        itog.update(kombo=" ".join(best["c"]), obj=best["objective"], obj_p1=V.obj(r1), delta=d, **V.po_godam(rb),
                    G4_osi=g4["ok"], G5_s9=bool(s9), proshla=bool(ok), verdikt="кандидат" if ok else "не прошла")
    K.write_table(OUT / "it10" / f"{sid}_itog.csv", [itog])
    K.save_ledger(sid, led, "ledger_it10.pkl")
    return itog
