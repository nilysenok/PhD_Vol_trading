# -*- coding: utf-8 -*-
"""etap1_poisk.py — драйвер поиска этапа 1 (Ж-047 п. 4): Моррис -> факторный план -> центр плато -> ворота Ж-044.

Поверх evaluate(params, period) -> dict (контракт строки — etap1_poisk_vorota.py). Фабрика (--fabrika модуль:функция)
по sid отдаёт (evaluate, s9_check(point) -> bool). Выход: docs/tables/zh047/S{k}_{morris,faktor,itog}.csv, только дополнение.
Запуск: python -m fastbt.etap1_poisk --strategii S1 S2 --fabrika fastbt.etap1:fabrika --jobs 2 [--morris-2019]
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fastbt import sweep_drivers as sd
from fastbt.etap1_poisk_vorota import Ledger, gates, matrix, obj, obj_morris, pbo, pick_best, psr_dsr, series, sigma126

SETKI = Path(__file__).parent / "designs" / "sweep"
OUT = Path("docs/tables/zh047")
SHAPKA = "# статус: разработка, train 2018–2019; validation не открыта; Ж-047\n"
POTOLOK, TOPK, MAX_JOBS = 5000, 20, 6
GATE_COLS = ("G1_pbo", "G2_sdelki", "G3_plato", "G5_s9", "G4_osi", "osi_pos", "ok")      # порядок ключей gates()


def load_setka(sid: str, d: Path = SETKI) -> dict:
    y = yaml.safe_load((d / f"zh047_{sid}.yaml").read_text(encoding="utf-8"))
    axes, yakor = {}, {}
    for blok, kl in (("signal", "ishodnoe"), ("obshchie", "yakor")):
        for k, v in y[blok].items():
            axes[k], yakor[k] = list(v["setka"]), v[kl]
            if yakor[k] not in axes[k]:
                raise ValueError(f"{sid}: якорь оси {k} вне сетки")
    return dict(sid=sid, axes=axes, yakor=yakor, r=y["morris"]["r"], seed=y["morris"]["seed"])


def choose_axes(ranked: list, axes: dict, potolok: int = POTOLOK) -> list:
    """Топ-5 по μ*; если куб > потолка — топ-4 (и ниже)."""
    for n in range(min(5, len(ranked)), 0, -1):
        if math.prod(len(axes[a]) for a in ranked[:n]) <= potolok:
            return ranked[:n]
    raise ValueError("ни одна ось не помещается в потолок")


def pick_center(led: Ledger, fact: dict, axes: dict, pb: float, s9_check) -> tuple:
    """Центр плато (среднее целевой по соседям), а не пик; затем ворота; ничья <= 0,05 — меньшая σ126 точки."""
    sc = sd.pick_by_plateau(fact, k=len(fact["rows"]), radius=1, min_neighbors=3)
    pt = lambda r: {k: r[k] for k in axes}                                       # noqa: E731
    cands = [r for r in sc if not r["edge"] and math.isfinite(r["objective"]) and not led(pt(r), "train").get("vyrozhdena")][:TOPK]
    for c in cands:
        c.update(sigma126=sigma126(led(pt(c), "train")), gates=gates(led, pt(c), axes, fact["keys"], pb, s9_check))
    return pick_best([c for c in cands if c["gates"]["ok"]], "plateau") or pick_best(cands, "plateau"), pt


def append_csv(path: Path, rows: list) -> None:
    """Только дополнение; колонки обязаны совпасть с шапкой файла."""
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        head = next(ln for ln in path.read_text(encoding="utf-8").splitlines() if not ln.startswith("#"))
        if head.split(",") != list(df.columns):
            raise ValueError(f"{path}: колонки не совпали с файлом — реестр только дополняется")
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        path.write_text(SHAPKA, encoding="utf-8")
        df.to_csv(path, mode="a", index=False)


def run_strategy(sid: str, evaluate, s9_check, out: Path = OUT, morris_2019: bool = False, run_id: str = "",
                 setka: dict | None = None) -> dict:
    S = setka or load_setka(sid)
    axes, led = S["axes"], Ledger(evaluate)
    mper = "ctrl2019" if morris_2019 else "train"
    mo = sd.morris_screen(led, axes, mper, n_paths=S["r"], seed=S["seed"], objective=obj_morris)
    fin = [(obj(res), p) for (_, per), (p, res) in led.cache.items() if per == mper and math.isfinite(obj(res))]
    best = max(fin, key=lambda t: t[0])[1] if fin else S["yakor"]
    plan = choose_axes(list(mo["effects"]), axes)
    base = {**S["yakor"], **{a: best[a] for a in ("sigma", "proverka") if a in axes and a not in plan}}
    fact = sd.factorial_sweep(led, base, {a: axes[a] for a in plan}, "train", objective=obj)
    M, n_mf = matrix(led), led.n_unique
    pb = pbo(M)
    c, pt = pick_center(led, fact, axes, pb, s9_check)
    nan = float("nan")
    res = led(pt(c), "train") if c else {}
    x = np.concatenate(list(series(res).values())) if c else None
    m = res.get("msr_by_year", {})
    row = dict(strategiya=sid, zapusk=run_id, morris_period=mper, osi_plana=" ".join(plan), N_morris=mo["n_evaluations"],
               N_faktor=fact["n_trials"], N_do_dosheta=n_mf, N_unikalnykh=led.n_unique, N_pbo=M.shape[1], pbo=pb,
               **{f"p_{k}": (pt(c)[k] if c else nan) for k in axes}, obj_min_msr=c["objective"] if c else nan,
               msr_2018=m.get(2018, nan), msr_2019=m.get(2019, nan), plateau=c["plateau"] if c else nan,
               sigma126=c["sigma126"] if c else nan, **{k: (c["gates"][k] if c else nan) for k in GATE_COLS},
               **(psr_dsr(x, M, led.n_unique) if c else dict(T=nan, psr=nan, dsr=nan, N_dsr=led.n_unique)),
               verdikt=("кандидат на validation" if c["gates"]["ok"] else "не дошла: ворота") if c else "не дошла: нет кандидата")
    append_csv(out / f"{sid}_morris.csv", [dict(strategiya=sid, zapusk=run_id, period=mper, os=a, rang=i, mu_star=e["mu_star"],
               sigma=e["sigma"], n=e["n"], v_plane=a in plan) for i, (a, e) in enumerate(mo["effects"].items(), 1)])
    append_csv(out / f"{sid}_faktor.csv", [dict(strategiya=sid, zapusk=run_id, **{k: r[k] for k in axes}, objective=r["objective"])
                                           for r in fact["rows"]])
    append_csv(out / f"{sid}_itog.csv", [row])
    return row


def _worker(a: tuple) -> dict:
    sid, fab, setki, kw = a
    mod, fn = fab.split(":")
    ev, s9 = getattr(importlib.import_module(mod), fn)(sid)
    return run_strategy(sid, ev, s9, setka=load_setka(sid, Path(setki)), **kw)


def main(argv=None) -> list:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategii", nargs="+", required=True)
    ap.add_argument("--fabrika", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--morris-2019", action="store_true", help="Моррис только на 2019, факторный — на обоих годах (Ж-047 п. 6)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--setki", default=str(SETKI))
    a = ap.parse_args(argv)
    kw = dict(out=Path(a.out), morris_2019=a.morris_2019, run_id=dt.datetime.now().isoformat(timespec="seconds"))
    with ProcessPoolExecutor(max_workers=max(1, min(a.jobs, MAX_JOBS))) as ex:
        return list(ex.map(_worker, [(s, a.fabrika, a.setki, kw) for s in a.strategii]))


if __name__ == "__main__":
    main()
