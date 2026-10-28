"""Итерация 10 Ж-051: сочетание прошедших элементов 2 (ансамбль), 3 (топ-k), 4 (сектор/ширина) — план «вкл/выкл».

Вход: ledger 2, 3, 4 (sha256 сверяется), таблицы `it{2,3,4}/{S}_itog.csv`. Кандидаты — сочетания из ≥ 2 элементов; соседи
сочетания — наборы, отличающиеся одним включённым элементом (Хэмминг 1), одиночные элементы и P1 — тоже. Выбор и ворота — как в
итерации 1 (Ж-051 п. 3): плато (среднее целевой по соседям), ничья ≤ 0,05 — меньшая σ126; G3 — ≥ 3 соседей с min mSR > 0;
G4 по k_s, R, τ — `gate_osi`. Соседей меньше 3 → «нет соседей», плато не пройдено. Общий PBO и N — по объединённой матрице
1 + 2 + 3 + 4 + 10.
"""
from __future__ import annotations

import datetime as dt
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fastbt import etap1, etap1_ansambl as A, etap1_poisk as P, etap1_poisk_vorota as V, zh051_kriteriy as K
from fastbt.etap1_eval import YAKOR
from fastbt.etap1_plus_eval import PlusEvaluator
from fastbt.zh051_iter import OUT
from fastbt.zh051_plato import GATES, PLUS_KEYS, fdr_of, ledger_cls, ledger_name

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
    """Кандидаты: сочетания из ≥ 2 прошедших элементов."""
    return [c for n in range(2, len(E) + 1) for c in itertools.combinations(E, n)]


def sosedi(c: tuple, E: list) -> list:
    """Наборы, отличающиеся от c одним элементом (включён/выключен); пустой набор — это P1."""
    return [tuple(x for x in E if (x in c) != (x == e)) for e in E]


def run_10(sid: str) -> dict:
    S, run_id = P.load_setka(sid), dt.datetime.now().isoformat(timespec="seconds")
    evr = PlusEvaluator(sid, etap1.build_inputs(sid), etap1.base_point(sid), 0.0004)
    row = pd.read_csv(Path("docs/tables/zh047") / f"{sid}_itog_v2.csv", comment="#").iloc[-1]
    p1 = {k: P._coerce(str(row[f"p_{k}"]), S["axes"][k]) for k in S["axes"]}
    E, sigpts = elementy(sid), None
    ens = E.get("ans")
    if ens:
        a = pd.read_csv(OUT / "it2" / f"{sid}_ansambl.csv", comment="#", dtype=str)
        osi = a[(a.nabor == ens["nabor"]) & (a.rezhim_ans == ens["rezhim_ans"])].iloc[-1]["osi"].split()
        sigpts = A.signal_points(A.tochki(p1, osi, S["axes"]), evr.sk.sig_names)

    def ev(p, per):                                        # точка с меткой ans — ансамбль вокруг P1, прочие оси — точки p
        q = {k: v for k, v in p.items() if k != "ans"}
        return evr.evaluate_ens({**YAKOR, **q}, sigpts, p["ans"].split("|")[1], per) if "ans" in p else evr.evaluate({**YAKOR, **q}, per)
    fdr = fdr_of(evr)
    cls = ledger_cls(fdr)
    led = K.load_ledger(sid, ledger_name(2, fdr), ev, cls=cls)
    for n in (3, 4):
        K.merge(led, K.load_ledger(sid, ledger_name(n, fdr), ev, cls=cls))
    if K.sverka_ledger(led, ev):
        raise SystemExit(f"{sid}: СТОП, строки ledger не воспроизведены текущим кодом")
    pb0, n0, names = V.pbo(V.matrix(led)), led.n_unique, list(E)

    def point(c):
        ext = {k: v for n in c if n != "ans" for k, v in E[n].items()}
        return {**p1, **ext, **({"ans": f"{ens['nabor']}|{ens['rezhim_ans']}"} if "ans" in c else {})}

    def res(c):
        if not c:
            return led(p1, "train")
        if c == ("ans",):
            return led.cache[(A.cfg_key(ens["nabor"], ens["rezhim_ans"]), "train")][1]
        return led(point(c), "train")
    cands = kombinacii(names)
    itog = dict(strategiya=sid, zapusk=run_id, iteraciya=10, N_do=n0, pbo_do=pb0, elementy=" ".join(names) or "нечего сочетать")
    rows = []
    for c in cands:
        nb = sosedi(c, names)
        o = [V.obj(res(x)) for x in (c, *nb)]
        r = res(c)
        rows.append(dict(c=c, nb=nb, objective=o[0], plateau=float(np.mean(o)) if all(map(math.isfinite, o)) else float("nan"),
                         n=len(o), sigma126=V.sigma126(r), edge=len(o) < 3,
                         adm=math.isfinite(o[0]) and V.dopustima(led, point(c), S["yakor"]) and not r.get("vyrozhdena")))
    M = V.matrix(led)
    pb, s9 = V.pbo(M), max(etap1.anchor_s9_diff(sid, evr).values()) <= 1e-12
    for c in rows:
        if not c["adm"]:
            continue
        pt, sdy = point(c["c"]), res(c["c"]).get("sdelok_y") or {}
        o = V.gate_osi(led, pt, S["axes"])
        g = dict(G1_pbo=bool(pb < V.PBO_MAX), G2_sdelki=bool(sdy) and all(v >= V.MIN_SDELOK for v in sdy.values()),
                 G3_plato=sum(1 for x in c["nb"] if V.obj(res(x)) > 0) >= 3, G4_osi=o["ok"], G5_s9=bool(s9))
        c["net_sosedey"] = len(c["nb"]) < 3
        c["gates"] = {**g, "ok": all(g[k] for k in GATES), "osi_pos": str(o["per"])}
    good = [c for c in rows if c.get("gates") and math.isfinite(c["plateau"])]
    best = V.pick_best([c for c in good if c["gates"]["ok"]], "plateau") or V.pick_best(good, "plateau")
    M = V.matrix(led)
    itog.update(pbo_obshchaya=V.pbo(M), N_obshchee=led.n_unique, N_pbo=M.shape[1], N_novyh=led.n_unique - n0)
    if best is None:
        itog.update(proshla=False, net_sosedey=True, verdikt="нечего сочетать" if len(names) < 2 else "не прошла: нет допустимой")
    else:
        d = best["objective"] - V.obj(res(()))
        g = best["gates"]
        ok = g["ok"] and math.isfinite(d) and d >= K.DELTA - 1e-12
        itog.update(kombo=" ".join(best["c"]), obj=best["objective"], obj_p1=V.obj(res(())), delta=d, plateau=best["plateau"],
                    **V.po_godam(res(best["c"])), **{k: g[k] for k in (*GATES, "osi_pos")}, net_sosedey=best["net_sosedey"],
                    proshla=bool(ok), verdikt="кандидат" if ok else ("не прошла: нет соседей, плато не пройдено"
                                                                     if best["net_sosedey"] else "не прошла"))
    K.write_table(OUT / "it10" / f"{sid}_itog.csv", [itog])
    K.save_ledger(sid, led, ledger_name(10, fdr))
    return itog
