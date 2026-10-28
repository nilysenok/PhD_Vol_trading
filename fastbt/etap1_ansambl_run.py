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
IT1_V1, OUT_V1, LED_V1 = Path("docs/tables/zh051_v1/it1"), Path("docs/tables/zh051_v1/it2"), Path("docs/tables/zh051/_ledger_v1")
REESTR = Path("docs/otbor/reestr_otbora_A.csv")
SHAPKA = "# статус: разработка, train 2018–2019; итерация 2 Ж-051; validation не открыта\n"
SHAPKA_V1 = "# статус: разработка, В1 train 2016–2019 (прогрев 2015); итерация 2 Ж-051; validation не открыта\n"


def paths(period: str = "train") -> tuple:
    """(таблицы it1, таблицы it2, каталог ledger). train — старые пути; В1 (train4) — zh051_v1 и _ledger_v1 (вне git)."""
    return (IT1, OUT, LED) if period == "train" else (IT1_V1, OUT_V1, LED_V1)


def restore(sid: str, evaluate, S: dict, period: str = "train") -> tuple:
    """(Ledger итерации 1, центр, ранжированные оси Морриса, строка itog_v2, N_до). В1: итог it1 — S{k}_itog_v2.csv."""
    it1 = paths(period)[0]
    row = pd.read_csv(it1 / f"{sid}_itog_v2.csv", comment="#").iloc[-1]
    old = pd.read_csv(it1 / f"{sid}_itog.csv", comment="#").iloc[-1] if period == "train" else row     # В1: один файл itog_v2
    axes, led = S["axes"], V.Ledger(evaluate)
    mper = old["morris_period"]
    mo = sd.morris_screen(led, axes, mper, n_paths=S["r"], seed=S["seed"], objective=V.obj_morris)
    f = pd.read_csv(it1 / f"{sid}_faktor.csv", comment="#", dtype=str)
    f = f[f.zapusk == old["zapusk"]]
    for _, r in f.iterrows():
        led({k: P._coerce(r[k], axes[k]) for k in axes}, period)
    rows = [{**{k: P._coerce(r[k], axes[k]) for k in axes}, "objective": float(r["objective"])} for _, r in f.iterrows()]
    n_mf = led.n_unique
    if n_mf != int(old["N_do_dosheta"]):
        raise SystemExit(f"{sid}: СТОП, Моррис+факторный дали {n_mf} ≠ N_do_dosheta {int(old['N_do_dosheta'])}")
    kf = {V.pkey({a: r[a] for a in axes}) for r in rows}
    plan = str(old["osi_plana"]).split()
    fact = dict(rows=rows, keys=plan, grid={a: axes[a] for a in plan})
    P.pick_center(led, fact, axes, float(old["pbo"]), lambda pt: True, S["yakor"], period)    # пробы плато идут в N
    fresh = V.Ledger(lambda p, per: led(p, per))                                          # как свежий Ledger в tolko_itog
    P.pick_center(fresh, fact, axes, float(old["pbo"]), lambda pt: True, S["yakor"], period)
    n_drv = int(old["N_unikalnykh"]) + len({k for k, _ in fresh.cache} - kf)
    if period != "train":
        n_drv = led.n_unique     # В1: итог it1 писал run_strategy — N уже после проб плато одним Ledger, без двойного счёта
    center = {k: P._coerce(str(row[f"p_{k}"]), axes[k]) for k in axes}
    # N в itog_v2 считал пробы по свежему Ledger без точек Морриса: проба, совпавшая с точкой Морриса, посчитана дважды.
    # Сверка: n_drv обязан равняться N itog_v2; честное число уникальных — led.n_unique (≤ n_drv).
    return led, center, list(mo["effects"]), row, n_drv


def save_ledger(sid: str, led: V.Ledger, name: str, period: str = "train") -> None:
    d = paths(period)[2] / sid
    d.mkdir(parents=True, exist_ok=True)
    with open(d / name, "wb") as fh:
        pickle.dump(led.cache, fh, protocol=4)


def write(path: Path, rows: list) -> None:
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(SHAPKA_V1 if "zh051_v1" in path.parts else SHAPKA, encoding="utf-8")
    df.to_csv(path, mode="a", index=False, header=not any(not ln.startswith("#") for ln in path.read_text("utf-8").splitlines()))


def dobavit(led: V.Ledger, nab: str, mode: str, res: dict, period: str) -> None:
    """Столбец конфигурации ансамбля в общую матрицу своего периода (A.dobavit_v_ledger пишет только под «train»)."""
    if period == "train":
        return A.dobavit_v_ledger(led, nab, mode, res)
    k = (A.cfg_key(nab, mode), period)
    if k in led.cache:
        raise ValueError(f"конфигурация {k[0]} уже в матрице: реестр только дополняется")
    led.cache[k] = ({"ansambl": nab, "rezhim_ansambl": mode}, res)


def run_sid(sid: str, period: str = "train") -> dict:  # xtask: длинная — линейный драйвер итерации, как до правки
    t0 = time.time()
    it2 = paths(period)[1]
    S = P.load_setka(sid)
    inp = etap1.build_inputs(sid) if period == "train" else etap1.build_inputs(sid, period=period)
    evr = AnsamblEvaluator(sid, inp, etap1.base_point(sid), 0.0004)
    ev = lambda p, per: evr.evaluate({**YAKOR, **p}, per)                      # noqa: E731
    led, center, ranked, it1, n_drv = restore(sid, ev, S, period)
    n0 = led.n_unique
    if n_drv != int(it1["N_unikalnykh"]) or n0 > n_drv:
        raise SystemExit(f"{sid}: СТОП, N воспроизведён {n_drv} ≠ N итерации 1 {int(it1['N_unikalnykh'])} (уникальных {n0})")
    res_c = led(center, period)
    ob = (lambda r: V.obj(r)) if period == "train" else (lambda r: V.obj(r, period))      # В1: min mSR по 4 годам, нехватка года -> nan
    if abs(ob(res_c) - float(it1["obj_min_msr"])) > 1e-9:
        raise SystemExit(f"{sid}: СТОП, центр не воспроизведён: {ob(res_c)} ≠ {it1['obj_min_msr']}")
    save_ledger(sid, led, "ledger_it1.pkl", period)
    pb1 = V.pbo(V.matrix(led, period))
    run_id, rows, cfg, s9r = dt.datetime.now().isoformat(timespec="seconds"), [], {}, {}
    for nab in A.NABORY:
        osi = A.osi_nabora(nab, ranked, S["axes"])
        pts = A.tochki(center, osi, S["axes"])
        sig = A.signal_points(pts, evr.sk.sig_names)
        s9r[nab] = sverka_s9(evr, sig)
        for mode in A.MODES:
            res = evr.evaluate_ens(center, sig, mode, period)
            dobavit(led, nab, mode, res, period)
            cfg[(nab, mode)] = res
            rows.append(dict(strategiya=sid, zapusk=run_id, nabor=nab, rezhim_ans=mode, osi=" ".join(osi), n_tochek=len(pts),
                             n_signal=len(sig), sverka_s9=str(s9r[nab]), objective=ob(res), **V.po_godam(res, period)))
    M = V.matrix(led, period)
    pb = V.pbo(M)
    base = V.storony(led(S["yakor"], "train")).get(2018)
    adm = {c: all(v >= 0.5 * base for v in V.storony(r).values()) and V.sdelok_ok(r["sdelok_y"], period) for c, r in cfg.items()}
    cand = [dict(c=c, o=ob(r), sigma126=V.sigma126(r), objective=ob(r)) for c, r in cfg.items() if adm[c]]
    pool = cand or [dict(c=c, o=ob(r), sigma126=V.sigma126(r), objective=ob(r)) for c, r in cfg.items()]
    best = V.pick_best(pool, "objective") or pool[0]
    c = best["c"]
    rb = cfg[c]
    plato = A.plato_ansambl({k: ob(r) for k, r in cfg.items()}, c)
    osi_c = V.gate_osi(led, center, S["axes"], period)
    s9_ok = s9r[c[0]]["rashozhdeniy"] == [] and max(etap1.anchor_s9_diff(sid, evr).values()) <= 1e-12
    g = dict(G1_pbo=bool(pb < V.PBO_MAX), G2_sdelki=V.sdelok_ok(rb["sdelok_y"], period),
             G3_plato_ansambl=plato["ok"], G4_osi_centra=osi_c["ok"], G5_s9=bool(s9_ok))
    ok = all(g.values())
    m1 = res_c["msr_by_year"]
    psr = V.psr_dsr(np.concatenate(list(V.series(rb).values())), M, led.n_unique)
    itog = dict(strategiya=sid, zapusk=run_id, nabor=c[0], rezhim_ans=c[1],
                obj_min_msr=ob(rb), **{f"it1_msr_{y}": m1[y] for y in V.PERIOD_YEARS[period]}, it1_obj=ob(res_c), pbo_it1=pb1,
                pbo_obshchaya=pb, N_it1_itog_v2=n_drv, N_it1_unikalnykh=n0, N_obshchee=led.n_unique, N_pbo=M.shape[1], **V.po_godam(rb, period), **g,
                plato_sosedey=plato["n_sosedey"], plato_osi_centra=str(osi_c["per"]), dopustima=adm[c], T=psr["T"],
                psr=psr["psr"], dsr=psr["dsr"], verdikt="кандидат на validation" if ok else "не дошла: ворота",
                vremya_s=round(time.time() - t0))
    write(it2 / f"{sid}_ansambl.csv", rows)
    write(it2 / f"{sid}_itog.csv", [itog])
    save_ledger(sid, led, "ledger_it2.pkl", period)
    if period != "train":
        return itog          # реестр отбора A — колонки train 2018–2019 (tune/ctrl): строки В1 в него не мешаем (Д7)
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
    ap.add_argument("--period", choices=("train", "train4"), default="train", help="train4 — В1: таблицы zh051_v1, ledger _ledger_v1")
    a = ap.parse_args(argv)
    with ProcessPoolExecutor(max_workers=max(1, min(a.jobs, 6))) as ex:
        return list(ex.map(run_sid, a.strategii, [a.period] * len(a.strategii)))


if __name__ == "__main__":
    main()
