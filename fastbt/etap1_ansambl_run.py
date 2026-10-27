"""Драйвер итерации 2 Ж-051 (ансамбль по плато): Ledger итерации 1 → 6 конфигураций → общая матрица, ворота, таблицы.

Запуск: FASTBT_MAIN=<своя копия> python -m fastbt.etap1_ansambl_run --strategii S1 ... --jobs 6
Ledger итерации 1 восстанавливается детерминированно (Моррис с тем же seed + строки S{k}_faktor.csv + пробы плато,
как `etap1_poisk.tolko_itog`); N обязан совпасть с S{k}_itog_v2.csv, иначе стоп. Ledger хранится в
docs/tables/zh051/_ledger/S{k}/ (pickle кэша; итерации 3–10 дополняют его). Только train 2018–2019.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from fastbt import etap1, etap1_ansambl as A, etap1_poisk as P, sweep_drivers as sd
from fastbt import etap1_poisk_vorota as V
from fastbt.etap1_ansambl_eval import AnsamblEvaluator, sverka_s9
from fastbt.etap1_eval import YAKOR

IT1, OUT, LED = Path("docs/tables/zh047"), Path("docs/tables/zh051/it2"), Path("docs/tables/zh051/_ledger")
REESTR = Path("docs/otbor/reestr_otbora_A.csv")
SHAPKA = "# статус: разработка, train 2018–2019; итерация 2 Ж-051; validation не открыта\n"


def restore(sid: str, evaluate, S: dict) -> tuple:
    """(Ledger итерации 1, центр, ранжированные оси Морриса, строка itog_v2, N_до)."""
    row = pd.read_csv(IT1 / f"{sid}_itog_v2.csv", comment="#").iloc[-1]
    old = pd.read_csv(IT1 / f"{sid}_itog.csv", comment="#").iloc[-1]
    axes, led = S["axes"], V.Ledger(evaluate)
    mper = old["morris_period"]
    mo = sd.morris_screen(led, axes, mper, n_paths=S["r"], seed=S["seed"], objective=V.obj_morris)
    f = pd.read_csv(IT1 / f"{sid}_faktor.csv", comment="#", dtype=str)
    f = f[f.zapusk == old["zapusk"]]
    for _, r in f.iterrows():
        led({k: P._coerce(r[k], axes[k]) for k in axes}, "train")
    rows = [{**{k: P._coerce(r[k], axes[k]) for k in axes}, "objective": float(r["objective"])} for _, r in f.iterrows()]
    n_mf = led.n_unique
    if n_mf != int(old["N_do_dosheta"]):
        raise SystemExit(f"{sid}: СТОП, Моррис+факторный дали {n_mf} ≠ N_do_dosheta {int(old['N_do_dosheta'])}")
    kf = {V.pkey({a: r[a] for a in axes}) for r in rows}
    plan = str(old["osi_plana"]).split()
    fact = dict(rows=rows, keys=plan, grid={a: axes[a] for a in plan})
    P.pick_center(led, fact, axes, float(old["pbo"]), lambda pt: True, S["yakor"])        # пробы плато идут в N
    fresh = V.Ledger(lambda p, per: led(p, per))                                          # как свежий Ledger в tolko_itog
    P.pick_center(fresh, fact, axes, float(old["pbo"]), lambda pt: True, S["yakor"])
    n_drv = int(old["N_unikalnykh"]) + len({k for k, _ in fresh.cache} - kf)
    center = {k: P._coerce(str(row[f"p_{k}"]), axes[k]) for k in axes}
    # N в itog_v2 считал пробы по свежему Ledger без точек Морриса: проба, совпавшая с точкой Морриса, посчитана дважды.
    # Сверка: n_drv обязан равняться N itog_v2; честное число уникальных — led.n_unique (≤ n_drv).
    return led, center, list(mo["effects"]), row, n_drv


def save_ledger(sid: str, led: V.Ledger, name: str) -> None:
    d = LED / sid
    d.mkdir(parents=True, exist_ok=True)
    with open(d / name, "wb") as fh:
        pickle.dump(led.cache, fh, protocol=4)


def write(path: Path, rows: list) -> None:
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(SHAPKA, encoding="utf-8")
    df.to_csv(path, mode="a", index=False, header=not any(not ln.startswith("#") for ln in path.read_text("utf-8").splitlines()))


def run_sid(sid: str) -> dict:
    t0 = time.time()
    S = P.load_setka(sid)
    inp = etap1.build_inputs(sid)
    evr = AnsamblEvaluator(sid, inp, etap1.base_point(sid), 0.0004)
    ev = lambda p, per: evr.evaluate({**YAKOR, **p}, per)                      # noqa: E731
    led, center, ranked, it1, n_drv = restore(sid, ev, S)
    n0 = led.n_unique
    if n_drv != int(it1["N_unikalnykh"]) or n0 > n_drv:
        raise SystemExit(f"{sid}: СТОП, N воспроизведён {n_drv} ≠ N итерации 1 {int(it1['N_unikalnykh'])} (уникальных {n0})")
    res_c = led(center, "train")
    if abs(V.obj(res_c) - float(it1["obj_min_msr"])) > 1e-9:
        raise SystemExit(f"{sid}: СТОП, центр не воспроизведён: {V.obj(res_c)} ≠ {it1['obj_min_msr']}")
    save_ledger(sid, led, "ledger_it1.pkl")
    pb1 = V.pbo(V.matrix(led))
    run_id, rows, cfg, s9r = dt.datetime.now().isoformat(timespec="seconds"), [], {}, {}
    for nab in A.NABORY:
        osi = A.osi_nabora(nab, ranked, S["axes"])
        pts = A.tochki(center, osi, S["axes"])
        sig = A.signal_points(pts, evr.sk.sig_names)
        s9r[nab] = sverka_s9(evr, sig)
        for mode in A.MODES:
            res = evr.evaluate_ens(center, sig, mode)
            A.dobavit_v_ledger(led, nab, mode, res)
            cfg[(nab, mode)] = res
            rows.append(dict(strategiya=sid, zapusk=run_id, nabor=nab, rezhim_ans=mode, osi=" ".join(osi), n_tochek=len(pts),
                             n_signal=len(sig), sverka_s9=str(s9r[nab]), objective=V.obj(res), **V.po_godam(res)))
    M = V.matrix(led)
    pb = V.pbo(M)
    base = V.storony(led(S["yakor"], "train")).get(2018)
    adm = {c: all(v >= 0.5 * base for v in V.storony(r).values()) and all(v >= V.MIN_SDELOK for v in r["sdelok_y"].values())
           for c, r in cfg.items()}
    cand = [dict(c=c, o=V.obj(r), sigma126=V.sigma126(r), objective=V.obj(r)) for c, r in cfg.items() if adm[c]]
    pool = cand or [dict(c=c, o=V.obj(r), sigma126=V.sigma126(r), objective=V.obj(r)) for c, r in cfg.items()]
    best = V.pick_best(pool, "objective") or pool[0]
    c = best["c"]
    rb = cfg[c]
    plato = A.plato_ansambl({k: V.obj(r) for k, r in cfg.items()}, c)
    osi_c = V.gate_osi(led, center, S["axes"])
    s9_ok = s9r[c[0]]["rashozhdeniy"] == [] and max(etap1.anchor_s9_diff(sid, evr).values()) <= 1e-12
    g = dict(G1_pbo=bool(pb < V.PBO_MAX), G2_sdelki=all(v >= V.MIN_SDELOK for v in rb["sdelok_y"].values()),
             G3_plato_ansambl=plato["ok"], G4_osi_centra=osi_c["ok"], G5_s9=bool(s9_ok))
    ok = all(g.values())
    m1 = res_c["msr_by_year"]
    psr = V.psr_dsr(np.concatenate(list(V.series(rb).values())), M, led.n_unique)
    itog = dict(strategiya=sid, zapusk=run_id, nabor=c[0], rezhim_ans=c[1],
                obj_min_msr=V.obj(rb), it1_msr_2018=m1[2018], it1_msr_2019=m1[2019], it1_obj=V.obj(res_c), pbo_it1=pb1,
                pbo_obshchaya=pb, N_it1_itog_v2=n_drv, N_it1_unikalnykh=n0, N_obshchee=led.n_unique, N_pbo=M.shape[1], **V.po_godam(rb), **g,
                plato_sosedey=plato["n_sosedey"], plato_osi_centra=str(osi_c["per"]), dopustima=adm[c], T=psr["T"],
                psr=psr["psr"], dsr=psr["dsr"], verdikt="кандидат на validation" if ok else "не дошла: ворота",
                vremya_s=round(time.time() - t0))
    write(OUT / f"{sid}_ansambl.csv", rows)
    write(OUT / f"{sid}_itog.csv", [itog])
    save_ledger(sid, led, "ledger_it2.pkl")
    reestr = [dict(strategiya=sid, klyuch=f"ansambl;nabor={r['nabor']};rezhim={r['rezhim_ans']};n_signal={r['n_signal']}",
                   rezhim="ansambl", istochnik="zh051_it2", tune_004=r["msr_2018"], ctrl_004=r["msr_2019"], storon_god=r["storon_2018"],
                   sosedey=plato["n_sosedey"] if (r["nabor"], r["rezhim_ans"]) == c else "", status="zh051_it2",
                   prichina="конфигурация ансамбля; 0,035 % не считалась", ekviv_s9=str(s9r[r["nabor"]]["sovpalo"]))
              for r in rows]
    with open(REESTR, "a", encoding="utf-8", newline="") as fh:                  # только дописывать
        pd.DataFrame(reestr).reindex(columns=pd.read_csv(REESTR, comment="#", nrows=0).columns).to_csv(fh, header=False, index=False)
    return itog


def main(argv=None) -> list:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategii", nargs="+", required=True)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args(argv)
    with ProcessPoolExecutor(max_workers=max(1, min(a.jobs, 6))) as ex:
        return list(ex.map(run_sid, a.strategii))


if __name__ == "__main__":
    main()
