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
import functools
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fastbt import sweep_drivers as sd
from fastbt.etap1_poisk_vorota import kw as _kw
from fastbt.etap1_poisk_vorota import (OSI, PERIOD_YEARS, Ledger, dopustima, gates, matrix, obj, obj_morris, pbo, pick_best, po_godam,
                                       pkey, psr_dsr, series, sigma126)

SETKI = Path(__file__).parent / "designs" / "sweep"
OUT = Path("docs/tables/zh047")
OUT_V1 = Path("docs/tables/zh051_v1/it1")        # В1 (train4): таблицы итерации 1; ledger В1 — docs/tables/zh051/_ledger_v1 (вне git)
SHAPKA = "# статус: разработка, train 2018–2019; validation не открыта; Ж-047\n"
SHAPKA_V1 = "# статус: разработка, В1 train 2016–2019 (прогрев 2015); итерация 1 Ж-051; validation не открыта\n"
POTOLOK, TOPK, MAX_JOBS = 5000, 20, 6
GATE_COLS = ("G1_pbo", "G2_sdelki", "G3_plato", "G5_s9", "G4_osi", "osi_pos", "kray_setki", "ok")      # порядок ключей gates()


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


def pick_center(led: Ledger, fact: dict, axes: dict, pb: float, s9_check, yakor: dict, period: str = "train") -> tuple:
    """Центр плато, а не пик; только допустимые по активности (dopustima); затем ворота;
    ничья <= 0,05 — меньшая σ126. Возвращает (кандидат | None, pt, число допустимых)."""
    sc = sd.pick_by_plateau(fact, k=len(fact["rows"]), radius=1, min_neighbors=3)
    pt = lambda r: {k: r[k] for k in axes}                                       # noqa: E731
    net_osi = lambda r: any(r[a] == "net" for a in OSI if a in fact["keys"])     # noqa: E731  «нет» на оси — край по Ж-048, не отсев
    ok = [r for r in sc if (not r["edge"] or net_osi(r)) and math.isfinite(r["objective"]) and not led(pt(r), period).get("vyrozhdena")
          and dopustima(led, pt(r), yakor, **_kw(period))]
    cands = ok[:TOPK]
    for c in cands:
        c.update(sigma126=sigma126(led(pt(c), period)), gates=gates(led, pt(c), axes, fact["keys"], pb, s9_check, **_kw(period)))
    return pick_best([c for c in cands if c["gates"]["ok"]], "plateau") or pick_best(cands, "plateau"), pt, len(ok)


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
        path.write_text(SHAPKA_V1 if "zh051_v1" in path.parts else SHAPKA, encoding="utf-8")
        df.to_csv(path, mode="a", index=False)


def itog_row(sid, run_id, mper, plan, axes, led, picked, head, psr_fn, plato_staroe=float("nan"), period: str = "train") -> dict:
    """Строка itog; без допустимой точки — те же колонки, поля точки nan. plato_staroe/novoe — ворота плато по осям (Ж-048)."""
    (c, pt, n_dop), nan = picked, float("nan")
    res = led(pt(c), period) if c else {}
    yr, m, ys = po_godam(res, period), res.get("msr_by_year", {}), PERIOD_YEARS[period]
    psr = psr_fn(np.concatenate(list(series(res).values()))) if c else dict(T=nan, psr=nan, dsr=nan, N_dsr=head["N_unikalnykh"])
    return dict(strategiya=sid, zapusk=run_id, morris_period=mper, osi_plana=" ".join(plan), **head,
                **{f"p_{k}": (pt(c)[k] if c else nan) for k in axes}, obj_min_msr=c["objective"] if c else nan,
                **{f"msr_{y}": m.get(y, nan) for y in ys}, **{f"sharpe_{y}": yr[f"sharpe_{y}"] for y in ys}, N_dopustimykh=n_dop, plateau=c["plateau"] if c else nan,
                sigma126=c["sigma126"] if c else nan, **{k: (c["gates"][k] if c else nan) for k in GATE_COLS}, **psr,
                plato_staroe=plato_staroe, plato_novoe=c["gates"]["G4_osi"] if c else nan,
                verdikt=("кандидат на validation" if c["gates"]["ok"] else "не дошла: ворота") if c else "не дошла: нет допустимой точки")


def _coerce(s: str, axis: list):
    """Значение из CSV-текста -> значение оси (1.0, 20, net, zakrytie)."""
    for v in axis:
        if str(v) == s or (not isinstance(v, str) and s.replace(".", "", 1).lstrip("-").isdigit() and float(s) == float(v)):
            return v
    raise ValueError(f"значение {s!r} вне оси {axis}")


def tolko_itog(sid: str, evaluate, s9_check, out: Path = OUT, run_id: str = "", setka: dict | None = None) -> dict:
    """Ж-048 п. 3: вердикт заново по готовым S{k}_faktor/itog.csv, без новой сетки; считаются только недостающие точки
    (активность, плато). itog дописывается; если шапка старого файла иная — в S{k}_itog_v2.csv. N_unikalnykh = старое + досчитанное (потолок)."""
    S = setka or load_setka(sid)
    axes, nan = S["axes"], float("nan")
    old = pd.read_csv(out / f"{sid}_itog.csv", comment="#").iloc[-1]
    plan = str(old["osi_plana"]).split()
    f = pd.read_csv(out / f"{sid}_faktor.csv", comment="#", dtype=str)
    f = f[f.zapusk == old["zapusk"]] if (f.zapusk == old["zapusk"]).any() else f[f.zapusk == f.zapusk.max()]
    rows = [{**{k: _coerce(r[k], axes[k]) for k in axes}, "objective": float(r["objective"])} for _, r in f.iterrows()]
    fact = dict(rows=rows, keys=plan, grid={a: axes[a] for a in plan})
    led = Ledger(evaluate)       # Ж-051: Ledger восстанавливается ДО досчёта (Моррис с тем же seed + строки faktor), иначе N завышен
    sd.morris_screen(led, axes, old["morris_period"], n_paths=S["r"], seed=S["seed"], objective=obj_morris)
    for r in rows:
        led({k: r[k] for k in axes}, "train")
    if led.n_unique != int(old["N_do_dosheta"]):
        raise ValueError(f"{sid}: Ledger не восстановлен: {led.n_unique} ≠ N_do_dosheta {int(old['N_do_dosheta'])}")
    picked = pick_center(led, fact, axes, float(old["pbo"]), s9_check, S["yakor"])
    head = {k: old[k] for k in ("N_morris", "N_faktor", "N_do_dosheta", "N_unikalnykh", "N_pbo", "pbo")}
    head["N_unikalnykh"] = led.n_unique         # все уникальные точки: Моррис + факторный + пробы плато, без двойного счёта
    psr = lambda x: dict(T=len(x), psr=nan, dsr=nan, N_dsr=head["N_unikalnykh"])        # noqa: E731  (справочно, нужен полный M)
    row = itog_row(sid, run_id, old["morris_period"], plan, axes, led, picked, head, psr, old.get("G4_osi", nan))
    p = out / f"{sid}_itog.csv"
    if p.exists() and next(ln for ln in p.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")).split(",") != list(row):
        p = out / f"{sid}_itog_v2.csv"
    append_csv(p, [row])
    return row


def popravka_N(sid: str, n: int, out: Path = OUT, run_id: str = "") -> dict:
    """Ж-051: допишет копию последней строки S{k}_itog_v2.csv с исправленным N (N_unikalnykh, N_dsr); старые строки не трогаются."""
    p = out / f"{sid}_itog_v2.csv"
    p = p if p.exists() else out / f"{sid}_itog.csv"
    row = pd.read_csv(p, comment="#").iloc[-1].to_dict()
    row.update(zapusk=run_id, N_unikalnykh=int(n), N_dsr=int(n))
    append_csv(p, [row])
    return row


def run_strategy(sid: str, evaluate, s9_check, out: Path | None = None, morris_2019: bool = False, run_id: str = "",
                 setka: dict | None = None, period: str = "train") -> dict:
    """period «train» — старый путь побайтно; «train4» (В1) — целевая min mSR по 4 годам, ворота и матрица по 4 годам, out = OUT_V1."""
    if period != "train" and morris_2019:
        raise ValueError("--morris-2019 только для train (Ж-047 п. 6)")
    out = out or (OUT if period == "train" else OUT_V1)
    ob, obm = (obj, obj_morris) if period == "train" else (functools.partial(obj, period=period), functools.partial(obj_morris, period=period))
    S = setka or load_setka(sid)
    axes, led = S["axes"], Ledger(evaluate)
    mper = "ctrl2019" if morris_2019 else period
    mo = sd.morris_screen(led, axes, mper, n_paths=S["r"], seed=S["seed"], objective=obm)
    fin = [(ob(res), p) for (_, per), (p, res) in led.cache.items() if per == mper and math.isfinite(ob(res))]
    best = max(fin, key=lambda t: t[0])[1] if fin else S["yakor"]
    plan = choose_axes(list(mo["effects"]), axes)
    base = {**S["yakor"], **{a: best[a] for a in ("sigma", "proverka") if a in axes and a not in plan}}
    fact = sd.factorial_sweep(led, base, {a: axes[a] for a in plan}, period, objective=ob)
    M, n_mf = matrix(led, period), led.n_unique
    pb = pbo(M)
    c, pt, n_dop = pick_center(led, fact, axes, pb, s9_check, S["yakor"], period)
    head = dict(N_morris=mo["n_evaluations"], N_faktor=fact["n_trials"], N_do_dosheta=n_mf, N_unikalnykh=led.n_unique,
                N_pbo=M.shape[1], pbo=pb)
    row = itog_row(sid, run_id, mper, plan, axes, led, (c, pt, n_dop), head, lambda x: psr_dsr(x, M, led.n_unique), period=period)
    append_csv(out / f"{sid}_morris.csv", [dict(strategiya=sid, zapusk=run_id, period=mper, os=a, rang=i, mu_star=e["mu_star"],
               sigma=e["sigma"], n=e["n"], v_plane=a in plan) for i, (a, e) in enumerate(mo["effects"].items(), 1)])
    append_csv(out / f"{sid}_faktor.csv", [dict(strategiya=sid, zapusk=run_id, **{k: r[k] for k in axes}, objective=r["objective"],
                                                **po_godam(led({k: r[k] for k in axes}, period), period)) for r in fact["rows"]])
    append_csv(out / (f"{sid}_itog.csv" if period == "train" else f"{sid}_itog_v2.csv"), [row])     # В1: имя как у zh047/itog_v2 (К-В1a)
    return row


def fabrika_v1(sid: str, period: str = "train4", commission: float = 0.0004):
    """Фабрика В1: входы периода (прогрев 2015), evaluate(params, period); якорь s9 — как в etap1.fabrika."""
    from fastbt import etap1
    evr = etap1.Evaluator(sid, etap1.build_inputs(sid, period=period), etap1.base_point(sid), commission)
    state = {}

    def s9(point: dict) -> bool:
        if "ok" not in state:
            state["ok"] = bool(max(etap1.anchor_s9_diff(sid, evr, commission).values()) <= 1e-12)
        return state["ok"]
    return (lambda params, per: evr.evaluate({**etap1.YAKOR, **params}, per)), s9


def _worker(a: tuple) -> dict:
    sid, fab, setki, kw = a
    mod, fn = fab.split(":")
    kw = dict(kw)
    per = kw.get("period", "train")
    ev, s9 = (getattr(importlib.import_module(mod), fn)(sid) if per == "train" else fabrika_v1(sid, per))
    if kw.pop("tolko_itog", False):
        kw.pop("morris_2019", None)
        if kw.pop("period", "train") != "train":
            raise ValueError("--tolko-itog только для train (Ж-048); В1 считает run_strategy")
        return tolko_itog(sid, ev, s9, setka=load_setka(sid, Path(setki)), **kw)
    return run_strategy(sid, ev, s9, setka=load_setka(sid, Path(setki)), **kw)


def main(argv=None) -> list:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategii", nargs="+", required=True)
    ap.add_argument("--fabrika", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--morris-2019", action="store_true", help="Моррис только на 2019, факторный — на обоих годах (Ж-047 п. 6)")
    ap.add_argument("--tolko-itog", action="store_true", help="новый вердикт по готовым faktor/itog без новой сетки (Ж-048 п. 3)")
    ap.add_argument("--period", choices=("train", "train4"), default="train", help="train4 — В1 (2016–2019, прогрев 2015)")
    ap.add_argument("--out", default=None, help="по умолчанию: train — docs/tables/zh047, train4 — docs/tables/zh051_v1/it1")
    ap.add_argument("--setki", default=str(SETKI))
    a = ap.parse_args(argv)
    kw = dict(out=Path(a.out or (OUT if a.period == "train" else OUT_V1)), period=a.period, morris_2019=a.morris_2019, tolko_itog=a.tolko_itog, run_id=dt.datetime.now().isoformat(timespec="seconds"))
    with ProcessPoolExecutor(max_workers=max(1, min(a.jobs, MAX_JOBS))) as ex:
        return list(ex.map(_worker, [(s, a.fabrika, a.setki, kw) for s in a.strategii]))


if __name__ == "__main__":
    main()
