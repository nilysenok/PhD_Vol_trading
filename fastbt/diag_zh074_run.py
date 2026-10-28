"""Ж-074, прогон: точки S1–S6 (шапки docs/tables/inv/S*_po_tikeram_0035.csv) -> сделки Tr -> zh074_dobor.csv, zh074_vnutri_dnya.csv.
Боевой путь не меняется: на время прогона обёртка над Run._enter только вешает на сделку σ входа и бары дня.
Запуск: FASTBT_MAIN=$PWD FASTBT_BARS1M=<D25/bars1m> python3 -m fastbt.diag_zh074_run"""
from __future__ import annotations

import ast
import csv
import subprocess
from contextlib import contextmanager

import numpy as np
import pandas as pd

from fastbt import diag_zh074 as z

COMMS = ((0.0004, "0,04"), (0.00035, "0,035"))      # 0,04 % — основная, 0,035 % — справочно


def point_of(sid: str) -> dict:
    head = open(z.INV / f"{sid}_po_tikeram_0035.csv", encoding="utf-8").readlines()[4]
    return ast.literal_eval(head.split("# точка: ", 1)[1].strip())


@contextmanager
def capture():
    """Вешает на сделку σ входа, бары дня входа, тик и признак лимитного входа; расчёт не меняет."""
    from fastbt import exec_e_run
    enter = exec_e_run.Run._enter

    def _enter(s, i, j, b, px, i0, sg, d_sig, sd=1):
        enter(s, i, j, b, px, i0, sg, d_sig, sd)
        s.tr.sg, s.tr.day, s.tr.tick, s.tr.lim = sg, s.days[i], s.p.tick, s.p.k is not None

    exec_e_run.Run._enter = _enter
    try:
        yield
    finally:
        exec_e_run.Run._enter = enter


def snapshot(tr, d, rate, comm: float) -> z.Tr:
    """Сделка движка -> Tr; d — даты тикера; ставка маржи — КС + 2 % на дату входа."""
    cal = int((d[tr.exit_day] - d[tr.entry_day]).astype("timedelta64[D]").astype(int))
    r = float(rate.asof(pd.Timestamp(str(d[tr.entry_day])[:10])))
    lo = z.last_ok_bar(len(tr.day.c), tr.exit_day == tr.entry_day, tr.exit_bar, tr.reason)
    return z.Tr(tr.side, tr.sg, tr.entry_px, tr.entry_bar, tr.day, lo, tr.exit_px, getattr(tr, "dcash", 0.0), comm, tr.tick, cal,
                r + 0.02, tr.reason, tr.lim)


def run_sk(sid: str, comm: float) -> dict:
    """Прогон точки Sk при комиссии comm (как inv_tikery.run_sk): закрытые сделки со входом ≥ 2018 + digest входов."""
    from fastbt import diag_zh069 as dz9
    from fastbt import etap1, msp
    from fastbt.diag_posle370 import accrue_dividends
    from fastbt.etap1_eval import YAKOR, Evaluator
    inp, rate = etap1.build_inputs(sid), msp.rate_cbr()
    evr = Evaluator(sid, inp, etap1.base_point(sid), comm)
    with accrue_dividends(), capture():
        rs = evr.excess({**YAKOR, **point_of(sid)}, "train")[4]
    trs, journal = [], 0.0
    for t, r in zip(evr.sk.tickers, rs):
        d = inp.meta[t]["dates"]
        s0 = int(np.searchsorted(d, np.datetime64("2018-01-01")))
        for tr in (x for x in r.trades if x.reason and x.entry_day >= s0):
            if not hasattr(tr, "day"):
                raise SystemExit(f"{sid}: сделка без бар-контекста (кэш прогона?)")
            q, jp = snapshot(tr, d, rate, comm), dz9.trade_pnl(tr)
            if abs(q.pnl - jp) > 1e-12:
                raise SystemExit(f"G2: {sid} {t}: PnL сделки {q.pnl} против журнала {jp}")
            trs.append(q)
            journal += jp
    return dict(trs=trs, journal=journal, digest=inp.digest())


def check_g2(sid: str, cl: str, res: dict) -> None:
    """G2: при L = 1 PnL по Sk = Σ журнала (1e-9) на всей сетке; Σ журнала = итог inv-таблицы (0,035 % или ITOG_004)."""
    for N in z.GRID_N:
        for x in z.GRID_X:
            v = z.pnl_at(res["trs"], N, x, 1.0)
            if abs(v - res["journal"]) > 1e-9:
                raise SystemExit(f"G2: {sid} {cl} N={N} x={x}: {v} против {res['journal']}")
    rows = [r for r in csv.DictReader(l for l in open(z.INV / f"{sid}_po_tikeram_0035.csv", encoding="utf-8") if not l.startswith("#"))]
    ref = float(rows[-1]["pnl_sum_%" if cl == "0,035" else "ITOG_004_pnl_sum_%"])
    if abs(100 * res["journal"] - ref) > 1e-4:
        raise SystemExit(f"G2: {sid} {cl}: Σ журнала {100 * res['journal']} % против inv-таблицы {ref} %")


def main() -> None:
    from fastbt.diag_zh066 import write_csv
    dob, day, dig = [], [], []
    for sid in ["S1", "S2", "S3", "S4", "S5", "S6"]:
        for comm, cl in COMMS:
            res = run_sk(sid, comm)
            check_g2(sid, cl, res)
            day += z.day_rows(sid, cl, res["trs"])
            dob += z.dobor_rows(sid, cl, res["trs"]) if res["trs"][0].lim else []
            print(sid, cl, len(res["trs"]), 100 * res["journal"], flush=True)
        dig.append(f"{sid}:{res['digest']}")
        if not res["trs"][0].lim:
            dob.append([sid, "-", "все", "не применяется (вход рынком)"] + [""] * (len(z.C_DOB) - 4))
    code = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=z.ROOT, capture_output=True, text=True).stdout.strip()
    meta = {"статус": z.STATUS, "коммит кода": code, "хэш входов": ";".join(dig), "комиссия": "0,04 % — основная, 0,035 % — справочно",
            "плата за заём на шорте": "нет (ряда нет)"}
    write_csv(z.OUT / "zh074_dobor.csv", meta, z.C_DOB, dob)
    write_csv(z.OUT / "zh074_vnutri_dnya.csv", meta, z.C_DAY, day)


if __name__ == "__main__":
    main()
