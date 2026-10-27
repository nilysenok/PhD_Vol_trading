"""Лестница сверки A «предзащита → сейчас», 2018–2021: по одному изменению за шаг → shag0_sverka_predzashchita.csv.

  SDF_CODE=<любое дерево с fastbt> FASTBT_MAIN=<копия> python3 sverka_A_lestnica.py <папка с R0/Q*.parquet> [--print-only]
Позиции — sverka_A_predzashchita.py pos (деревья кода до/после #259); слой портфеля — portfolio() оттуда.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sverka_A_predzashchita as S  # noqa: E402

TARGET = -0.8767          # shag0_A_metriki.csv, A, 0,04 %, после аудита (PR #264)
OUT = Path(__file__).resolve().parents[2] / "docs" / "tables" / "strat_dev_v3" / "shag0_sverka_predzashchita.csv"
BASE = dict(mask1=True, sel=0.004, corp=False, tree="old", tf=False,
            comm=0.0, rate="back", fwd=False, exit_fee=False, lev_cal=False, agg="pairs")
RUNS = {(True, .004, False, "old", False): "R0", (True, .0005, False, "old", False): "Q1",
        (True, .0005, True, "old", False): "Q1c", (False, .0005, False, "old", False): "Q2",
        (False, .0005, True, "old", False): "Q3", (False, .0005, True, "new", False): "Q4",
        (False, .0005, True, "new", True): "Q5", (False, .004, False, "old", False): "P1",(False, .0004, True, "new", True): "R5"}
STEPS = [  # (код, текст, изменение состояния | None — не выполнено/вклад 0 по построению, пояснение)
    ("б1", "комиссия отбора 0,40 % → 0,05 % (так в кэше A: main.yaml comm; не 0,04)", dict(sel=.0005), ""),
    ("а", "критерий отбора: без ставки → избыточный над ставкой", None, "не выполнено: нужен флаг в stats.window (движок, запрещено)"),
    ("б2", "комиссия ряда 0 → 0,04 %", dict(comm=.0004), ""),
    ("в1", "ставка: ряд из CSV (back) → решения ЦБ", dict(rate="cbr"), ""),
    ("в2", "Н2: ставка за календарные дни [d, d_next)", dict(fwd=True), ""),
    ("г", "Н5: комиссия выхода позиции на 30.12.2021", dict(exit_fee=True), ""),
    ("д", "Н4: плата за плечо по календарным дням", dict(lev_cal=True), "экспозиция A ≤ 1: плечо не возникает, вклад 0 по построению"),
    ("е", "состав сигналов S1–S6", None, "те же S1…S6 — вклад 0"),
    ("ж", "corporate actions: без → с corporate_actions.csv", dict(corp=True), ""),
    ("з", "тикеры: 17 те же", None, "те же 17 — вклад 0"),
    ("и1", "утечка окна: маска [1] → [0]", dict(mask1=False), ""),
    ("и2", "агрегирование: 1/102 пар → 17 тикеров поровну", dict(agg="tickw"), "издержки по парам, вес 1/17 на тикер; при полной панели 17×6 равны (зачёт тикеров с неттингом позиций — иное определение, вне лестницы)"),
    ("и3", "окна отбора платят выход (#259)", dict(tree="new"), ""),
    ("и4", "фильтр оборота (main.yaml: turnover_filter)", dict(tf=True), "в печатном A его, видимо, не было"),
]


def run_name(st):
    return RUNS.get((st["mask1"], st["sel"], st["corp"], st["tree"], st["tf"]))


def evaluate(d: Path, st: dict, cache: dict) -> tuple:
    n = run_name(st)
    if n is None:
        return None
    if n not in cache:
        cache[n] = pd.read_parquet(d / f"{n}.parquet")
    p = S.portfolio(cache[n], S.load_rate(st["rate"]), st["comm"], st["agg"], st["fwd"], st["exit_fee"], st["lev_cal"])
    return S.sharpe(p["excess"]), S.by_year(p), n


def main() -> None:
    d = Path(sys.argv[1])
    cache, rows = {}, []
    st = dict(BASE)
    sh0, y0, n0 = evaluate(d, st, cache)
    prev, prev_y = sh0, y0

    def row(code, text, sh, y, run, note):
        r = dict(shag=code, izmenenie=text, run=run, sharpe=sh, d_sharpe=sh - prev if rows else np.nan, zametka=note)
        for k in S.YEARS:
            r[f"sharpe_{k}"] = y[k]
            r[f"d_{k}"] = y[k] - prev_y[k] if rows else np.nan
        return r
    rows.append(row("0", "A как на предзащите: маска [1], отбор 0,40 %, ряд без комиссии, ставка back, 1/102", sh0, y0, n0, ""))
    deltas = {}
    for code, text, chg, note in STEPS:
        if chg:
            st.update(chg)
        sh, y, n = evaluate(d, st, cache)
        rows.append(row(code, text, sh, y, n, note))
        deltas[code] = (sh - prev, chg)
        prev, prev_y = sh, y
    rest = TARGET - prev
    rows.append(dict(shag="к", izmenenie="остаток: Шарп «сейчас» (−0,8767, PR #264) − последний шаг", sharpe=TARGET, d_sharpe=rest,
                     zametka="ожидается ≈ 0"))
    # порядок: самый крупный шаг, поставленный первым (если нужная пара позиций посчитана)
    for code, (dl, chg) in sorted(deltas.items(), key=lambda kv: -abs(kv[1][0])):
        if not chg:
            continue
        s1 = dict(BASE)
        s1.update(chg)
        e = evaluate(d, s1, cache)
        if e is None:
            continue
        sg = "знак меняется" if np.sign(e[0] - sh0) != np.sign(dl) and abs(dl) > 1e-9 else "знак тот же"
        rows.append(dict(shag=f"порядок: {code} первым", izmenenie="вклад шага, поставленного первым", run=e[2], sharpe=e[0],
                         d_sharpe=e[0] - sh0, zametka=f"в лестнице {dl:+.4f}; {sg}"))
        break
    # чувствительность: отбор 0,04 % вместо 0,05 % (задание) на финальном слое
    fin = dict(st, sel=.0004)
    e = evaluate(d, fin, cache)
    if e:
        rows.append(dict(shag="чувств.: отбор 0,04 %", izmenenie="последняя строка с комиссией отбора 0,04 %", run=e[2],
                         sharpe=e[0], d_sharpe=e[0] - prev, zametka="задание: 0,04; в кэше A — 0,05"))
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, sep=";", decimal=",", index=False, float_format="%.4f", encoding="utf-8-sig")
    print(df[["shag", "run", "sharpe", "d_sharpe", "d_2018", "d_2019", "d_2020", "d_2021"]].to_string(float_format=lambda v: "%.4f" % v))


if __name__ == "__main__":
    main()
