"""Драйвер итераций 3, 4 и 10 Ж-051 (быстрый вариант, Ж-054).

Запуск: FASTBT_MAIN=<своя копия> python -m fastbt.zh051_iter --it 3|4|10 --strategii S1 ... [--jobs 6]
Итерации 3 и 4 стартуют от `ledger_it2.pkl` (sha256 сверяется до использования), пишут `ledger_it{N}.pkl` и строку в
SHA256SUMS.txt. Итерация 10 объединяет ledger 3 и 4 и сочетает прошедшие элементы 2, 3, 4 (вкл/выкл). N и PBO — по общей
матрице ledger; контроли в него не входят. В конце — правило остановки Ж-054. Только train 2018–2019.
"""
from __future__ import annotations

import argparse
import datetime as dt
import functools
import itertools
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fastbt import etap1, etap1_ansambl as A, etap1_poisk as P, etap1_poisk_vorota as V, zh051_kriteriy as K
from fastbt.etap1_eval import NET, YAKOR
from fastbt.etap1_plus_eval import PlusEvaluator

SETKI, OUT = Path("fastbt/designs/sweep"), Path("docs/tables/zh051")
PLUS_KEYS = ("top_k", "sektor_trend", "shirina", "ma_n", "porog_shiriny")


def norm(ext: dict):
    """Оси итерации → каноническая точка; всё выключено — None (= P1); shirina = net → порог не влияет."""
    e = dict(ext)
    if e.get("sektor_trend", NET) == NET and e.get("shirina", NET) == NET and e.get("top_k", NET) == NET:
        return None
    if e.get("shirina", NET) == NET and "porog_shiriny" in e:
        e["porog_shiriny"] = 0.5
    return e


def plan(y: dict) -> tuple:
    """(точки плана без P1, сетки осей) из zh051_it{N}.yaml."""
    grid = {a: list(v["setka"]) for a, v in y["osi"].items()}
    out, seen = [], set()
    for vals in itertools.product(*grid.values()):
        e = norm(dict(zip(grid, vals)))
        if e is not None and tuple(sorted(e.items())) not in seen:
            seen.add(tuple(sorted(e.items())))
            out.append(e)
    return out, grid


def plateau(led, p1: dict, e: dict, grid: dict) -> float:
    """Среднее obj точки и её соседей ±1 по осям итерации (сосед-выключено = P1); любой nan → nan."""
    pts = [e] + [norm(q) for q in V.neighbors({a: e[a] for a in grid if a in e}, [a for a in grid if a in e], grid)]
    v = [V.obj(led({**p1, **(q or {})}, "train")) for q in pts]
    return float(np.mean(v)) if all(math.isfinite(x) for x in v) else float("nan")


def kontroli(evr, pt: dict, it: int, y: dict) -> list:
    """20 контролей вне Ledger: it3 — перемешанные баллы, it4 — циклический сдвиг рядов ≥ 63 дней (к2)."""
    kc = y["kontrol"]
    rng = np.random.default_rng(int(kc["seed"]))
    lags = rng.choice(np.arange(kc.get("lag_min", 63), 252), size=K.N_KONTROL, replace=False)
    ks = [("sluchaynyy_rang", kc["seed"] + i) if it == 3 else ("krug_sdvig", int(lags[i])) for i in range(K.N_KONTROL)]
    return [V.obj(evr.evaluate({**YAKOR, **pt, "kontrol": k}, "train")) for k in ks]


def run_it(sid: str, it: int) -> dict:
    y = yaml.safe_load((SETKI / f"zh051_it{it}.yaml").read_text("utf-8"))
    S, run_id = P.load_setka(sid), dt.datetime.now().isoformat(timespec="seconds")
    evr = PlusEvaluator(sid, etap1.build_inputs(sid), etap1.base_point(sid), 0.0004)
    ev = lambda p, per: evr.evaluate({**YAKOR, **p}, per)                    # noqa: E731
    led = K.load_ledger(sid, "ledger_it2.pkl", ev)
    if K.sverka_ledger(led, ev):
        raise SystemExit(f"{sid}: СТОП, строки ledger не воспроизведены текущим кодом")
    row = pd.read_csv(Path("docs/tables/zh047") / f"{sid}_itog_v2.csv", comment="#").iloc[-1]
    p1 = {k: P._coerce(str(row[f"p_{k}"]), S["axes"][k]) for k in S["axes"]}
    pb0, n0, keys0 = V.pbo(V.matrix(led)), led.n_unique, set(led.cache)
    pts, grid = plan(y)
    r1 = led(p1, "train")
    rows = []
    for e in pts:
        r = led({**p1, **e}, "train")
        rows.append(dict(e=e, obj=V.obj(r), plateau=plateau(led, p1, e, grid), sigma126=V.sigma126(r),
                         adm=V.dopustima(led, {**p1, **e}, S["yakor"]) and not r.get("vyrozhdena")))
    best = V.pick_best([c for c in rows if c["adm"] and math.isfinite(c["plateau"])], "plateau")
    itog = dict(strategiya=sid, zapusk=run_id, iteraciya=it, N_do=n0, pbo_do=pb0)
    if best is None:
        itog.update(proshla=False, verdikt="не прошла: нет допустимой точки")
    else:
        itog.update(_ocenit(evr, led, S, y, p1, r1, best, grid, pb0, it))
    M = V.matrix(led)
    new = [k for k in led.cache if k not in keys0]
    cols = [np.r_[V.series(led.cache[k][1])[2018], V.series(led.cache[k][1])[2019]] for k in new
            if {2018, 2019} <= set(V.series(led.cache[k][1]))]
    itog.update(pbo_obshchaya=V.pbo(M), pbo_tolko_it=V.pbo(np.column_stack(cols)) if len(cols) > 1 else float("nan"),
                N_obshchee=led.n_unique, N_pbo=M.shape[1], N_novyh=led.n_unique - n0)
    K.write_table(OUT / f"it{it}" / f"{sid}_plan.csv",
                  [{"strategiya": sid, **c["e"], **{k: c[k] for k in ("obj", "plateau", "sigma126", "adm")}} for c in rows])
    K.write_table(OUT / f"it{it}" / f"{sid}_itog.csv", [itog])
    K.save_ledger(sid, led, f"ledger_it{it}.pkl")
    return itog


def _ocenit(evr, led, S, y, p1, r1, best, grid, pb0, it) -> dict:
    """Выбранная Pi → ворота G4/s9, контроли, критерий §3."""
    e, pi = best["e"], {**p1, **best["e"]}
    ri = led(pi, "train")
    nb = [norm(q) for q in V.neighbors({a: e[a] for a in grid if a in e}, [a for a in grid if a in e], grid)]
    g4, g4_p1 = V.gate_osi(led, pi, S["axes"]), V.gate_osi(led, p1, S["axes"])
    s9 = max(etap1.anchor_s9_diff(evr.sid, evr).values()) <= 1e-12
    pbo_after = V.pbo(V.matrix(led))
    ctrl = kontroli(evr, pi, it, y)
    c = K.kriteriy(dict(tsel_pi=V.obj(ri), tsel_p1=V.obj(r1), msr_pi=ri["msr_by_year"], msr_p1=r1["msr_by_year"],
                        nb_tsel=[V.obj(led({**p1, **q}, "train")) for q in nb if q is not None], sdelok_pi=ri["sdelok_y"],
                        s9_ok=s9, g4_ok=g4["ok"] or not g4_p1["ok"], pbo_do=pb0, pbo_posle=pbo_after, kontrol=ctrl))
    psr = V.psr_dsr(np.concatenate(list(V.series(ri).values())), V.matrix(led), led.n_unique)
    return dict(**{f"pi_{a}": e.get(a, "") for a in PLUS_KEYS}, obj_pi=V.obj(ri), obj_p1=V.obj(r1), **V.po_godam(ri), **c,
                G1_pbo=bool(pbo_after < V.PBO_MAX), G4_osi=g4["ok"], G5_s9=bool(s9), psr=psr["psr"], dsr=psr["dsr"],
                verdikt="прошла" if c["proshla"] else "не прошла")


def vorota(r: dict) -> bool:
    return bool(r.get("proshla")) and bool(r.get("G1_pbo")) and bool(r.get("G4_osi")) and bool(r.get("G5_s9"))


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--it", type=int, required=True, choices=(3, 4, 10))
    ap.add_argument("--strategii", nargs="+", required=True)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args(argv)
    if a.it == 10:
        from fastbt.zh051_iter10 import run_10 as fn
    else:
        fn = functools.partial(run_it, it=a.it)
    with ProcessPoolExecutor(max_workers=max(1, min(a.jobs, 6))) as ex:
        res = list(ex.map(fn, a.strategii))
    if len(res) < 6:
        return dict(stop=None, itog="правило остановки не проверяется: считаны не все 6 стратегий")
    st = K.ostanovka({r["strategiya"]: r["pbo_obshchaya"] for r in res}, {r["strategiya"]: vorota(r) for r in res})
    K.write_table(OUT / f"it{a.it}" / "ostanovka.csv", [{**st, "iteraciya": a.it, "dalshe": " ".join(st["dalshe"])}])
    print(st["itog"])
    return st


if __name__ == "__main__":
    main()
