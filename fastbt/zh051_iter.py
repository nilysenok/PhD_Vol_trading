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
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fastbt import etap1, etap1_ansambl as A, etap1_poisk as P, etap1_poisk_vorota as V, zh051_kriteriy as K
from fastbt.etap1_eval import YAKOR
from fastbt.etap1_plus_eval import PlusEvaluator
from fastbt.zh051_plato import PLUS_KEYS, fdr_of, ledger_cls, ledger_name, norm, vybor

SETKI, OUT = Path("fastbt/designs/sweep"), Path("docs/tables/zh051")


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
    led = K.load_ledger(sid, ledger_name(2, fdr_of(evr)), ev, cls=ledger_cls(fdr_of(evr)))
    if K.sverka_ledger(led, ev):
        raise SystemExit(f"{sid}: СТОП, строки ledger не воспроизведены текущим кодом")
    row = pd.read_csv(Path("docs/tables/zh047") / f"{sid}_itog_v2.csv", comment="#").iloc[-1]
    p1 = {k: P._coerce(str(row[f"p_{k}"]), S["axes"][k]) for k in S["axes"]}
    pb0, n0, keys0 = V.pbo(V.matrix(led)), led.n_unique, set(led.cache)
    _, grid = plan(y)
    r1, s9 = led(p1, "train"), max(etap1.anchor_s9_diff(sid, evr).values()) <= 1e-12
    best, rows = vybor(led, p1, grid, S["axes"], S["yakor"], s9)
    itog = dict(strategiya=sid, zapusk=run_id, iteraciya=it, N_do=n0, pbo_do=pb0)
    if best is None:
        itog.update(proshla=False, net_sosedey=True, verdikt="не прошла: нет допустимой точки")
    else:
        itog.update(_ocenit(evr, led, y, p1, r1, best, pb0, it))
    M = V.matrix(led)
    new = [k for k in led.cache if k not in keys0]
    cols = [np.r_[V.series(led.cache[k][1])[2018], V.series(led.cache[k][1])[2019]] for k in new
            if {2018, 2019} <= set(V.series(led.cache[k][1]))]
    itog.update(pbo_obshchaya=V.pbo(M), pbo_tolko_it=V.pbo(np.column_stack(cols)) if len(cols) > 1 else float("nan"),
                N_obshchee=led.n_unique, N_pbo=M.shape[1], N_novyh=led.n_unique - n0)
    K.write_table(OUT / f"it{it}" / f"{sid}_plan.csv",
                  [{"strategiya": sid, **c["e"], **{k: c[k] for k in ("objective", "plateau", "n", "sigma126", "adm", "edge")}}
                   for c in rows])
    K.write_table(OUT / f"it{it}" / f"{sid}_itog.csv", [itog])
    K.save_ledger(sid, led, ledger_name(it, fdr_of(evr)))
    return itog


def _ocenit(evr, led, y, p1, r1, best, pb0, it) -> dict:
    """Выбранная Pi: ворота плато (iteration-1: G1–G5, пересчитаны в `vybor`), контроли, критерий §3 (П1, П2, П4, П5)."""
    e, g = best["e"], best["gates"]
    pi = {**p1, **e}
    ri = led(pi, "train")
    ctrl = kontroli(evr, pi, it, y)
    pbo_after = V.pbo(V.matrix(led))
    c = K.kriteriy(dict(tsel_pi=V.obj(ri), tsel_p1=V.obj(r1), msr_pi=ri["msr_by_year"], msr_p1=r1["msr_by_year"], nb_tsel=[],
                        plato_ok=bool(g["G3_plato"]), sdelok_pi=ri["sdelok_y"], s9_ok=g["G5_s9"], g4_ok=g["G4_osi"],
                        pbo_do=pb0, pbo_posle=pbo_after, kontrol=ctrl))
    psr = V.psr_dsr(np.concatenate(list(V.series(ri).values())), V.matrix(led), led.n_unique)
    ok = c["proshla"] and g["ok"]
    return dict(**{f"pi_{a}": e.get(a, "") for a in PLUS_KEYS}, obj_pi=V.obj(ri), obj_p1=V.obj(r1), **V.po_godam(ri),
                **{k: v for k, v in c.items() if k != "proshla"},
                **{k: g[k] for k in ("G1_pbo", "G2_sdelki", "G3_plato", "G4_osi", "G5_s9", "osi_pos", "n_sosedey")},
                net_sosedey=best["net_sosedey"], psr=psr["psr"], dsr=psr["dsr"], proshla=bool(ok),
                verdikt="прошла" if ok else ("не прошла: нет соседей, плато не пройдено" if best["net_sosedey"] else "не прошла"))


PLATA_SHORTA = (0.0, 0.05, 0.10, 0.15)       # годовых на шортовую ногу; 0 — основная колонка, остальные справочно (К-26)


def sdvig_sigma(tickers: list) -> str:
    """Д1 для σ̂ (ewma094) на 2018–2019: corr(Δσ², RV_{d+k}), k = −2…+1, по тикерам; пик обязан быть на k = 0."""
    from fastbt import sigma_real as SR
    rv, res = SR.load_rv(list(tickers)), []
    for t, g in rv.groupby("ticker"):
        r = g.set_index("date")["rv"].sort_index()
        s, r = SR.sigma_all(r)["ewma094"], r[r.index < "2020-01-01"]
        res.append(SR.proverka_sdvig(s.loc[r.index][r.index >= "2018-01-01"], r[r.index >= "2018-01-01"]))
    ok = sum(x["ok"] for x in res)
    return f"Д1 σ̂ ewma094, 2018–2019: пик на k=0 у {ok} из {len(res)} тикеров -> " + ("пройдена" if ok == len(res) else "НЕ ПРОЙДЕНА")


def run_26(sid: str) -> list:
    """Итерация 26: «только лонг» — одна конфигурация от P1 (`storona=long`), в ledger идёт только она; плата за шорт — справочно
    (вне ledger, N не растёт). Стартует от ledger_it10 (вся матрица 1–10); пишет ledger_it26 один раз."""
    S = P.load_setka(sid)
    evr = PlusEvaluator(sid, etap1.build_inputs(sid), etap1.base_point(sid), 0.0004)
    ev = lambda p, per: evr.evaluate({**YAKOR, **p}, per)                    # noqa: E731
    led = K.load_ledger(sid, ledger_name(10, fdr_of(evr)), ev, cls=ledger_cls(fdr_of(evr)))
    stale = bool(K.sverka_ledger(led, ev))        # ledger_it10 посчитан до #370 (02:46 < 09:47): строки не воспроизводятся
    row = pd.read_csv(Path("docs/tables/zh047") / f"{sid}_itog_v2.csv", comment="#").iloc[-1]
    p1 = {k: P._coerce(str(row[f"p_{k}"]), S["axes"][k]) for k in S["axes"]}
    nan, n0 = float("nan"), led.n_unique
    call = ev if stale else led                                                # устаревшая матрица не смешивается с новым движком
    r_obe = call(p1, "train")
    pb0 = nan if stale else V.pbo(V.matrix(led))
    r_long = call({**p1, "storona": "long"}, "train")
    pb1, out = (nan if stale else V.pbo(V.matrix(led))), []
    for st, pl, r, pb in [("obe", 0.0, r_obe, pb0), ("long", 0.0, r_long, pb1)] + [
            ("obe", f, ev({**p1, "plata_shorta": f}, "train"), nan) for f in PLATA_SHORTA[1:]]:
        out.append(dict(strategiya=sid, storona=st, plata_shorta=pl, osnovnaya=pl == 0.0, **V.po_godam(r), obj=V.obj(r),
                        pbo_obshchaya=pb, N_obshchee=nan if stale else led.n_unique, N_novyh=nan if stale else led.n_unique - n0,
                        matrica="устарела: ledger до #370, PBO не считан" if stale else "ok", vhody=evr.inp.digest()[:16]))
    path = OUT / "it26" / f"{sid}_itog.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    path.write_text(f"# статус: разработка 2018–2019, после аудита, после дивидендов; Ж-069 п. 3; validation не открыта\n"
                    f"# коммит кода: {sha}; хэш входов: {out[0]['vhody']}; комиссия 0,04 %; плата за шорт — годовых, справочно\n", "utf-8")
    pd.DataFrame(out).to_csv(path, mode="a", index=False)
    if not stale:
        K.save_ledger(sid, led, ledger_name(26, fdr_of(evr)))                            # устаревшую матрицу не размножаем
    return out


def vorota(r: dict) -> bool:
    return bool(r.get("proshla")) and all(bool(r.get(k)) for k in ("G1_pbo", "G2_sdelki", "G3_plato", "G4_osi", "G5_s9"))


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--it", type=int, required=True, choices=(3, 4, 10, 26))
    ap.add_argument("--strategii", nargs="+", required=True)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args(argv)
    if a.it == 26:
        print(sdvig_sigma(etap1.build_inputs(a.strategii[0]).sk.tickers))
        with ProcessPoolExecutor(max_workers=max(1, min(a.jobs, 3))) as ex:
            return dict(rows=list(ex.map(run_26, a.strategii)))
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
