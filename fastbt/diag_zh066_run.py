"""Ж-066 п. 1, прогон: точка итерации 1 -> сделки с силой сигнала, заявки (выставлена/исполнена), дневной ряд портфеля.
Движок не меняется: Evaluator.excess, поверх на время прогона обёртки над Run._place/_try_order только пишут журнал заявок
(расчёт не трогают; сверка PnL с S{k}_sdelki.csv — в diag_zh066.reconcile). Диагностика, не выбор; train 2018–2019;
без дивидендов. Запуск: FASTBT_MAIN=$PWD python3 -m fastbt.diag_zh066_run S1 [выход.pkl]"""
from __future__ import annotations

import ast
import csv
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from fastbt import etap1, exec_e_run
from fastbt.etap1_eval import YAKOR, Evaluator

ROOT = Path(__file__).resolve().parents[1]


def point(sid: str) -> dict:
    """Точка итерации 1 — шапка `# точка:` журнала docs/tables/diag/{sid}_sdelki.csv (то, что видел владелец)."""
    head = open(ROOT / f"docs/tables/diag/{sid}_sdelki.csv", encoding="utf-8").readlines()[1]
    return ast.literal_eval(head.split("# точка: ", 1)[1].rsplit("; стороны-сделки", 1)[0])


@contextmanager
def log_orders(sink: list):
    """Пишет в sink по записи на заявку: тикер, день сигнала, сторона, уровень, цена решения, исполнена ли, цена."""
    place, try_order = exec_e_run.Run._place, exec_e_run.Run._try_order

    def _place(s, close, sd, sg, j, i, **kw):
        place(s, close, sd, sg, j, i, **kw)
        s.order["close_d"] = close

    def _try(s, i, j, d):
        o, nf = s.order, s.n_fills
        try_order(s, i, j, d)
        hit = s.n_fills > nf
        sink.append(dict(tiker=s.t, d_sig=o["d_sig"], side=o["sd"], L=o["L"], close_d=o["close_d"], isp=hit,
                         px=s.trades[-1].entry_px if hit else float("nan")))

    exec_e_run.Run._place, exec_e_run.Run._try_order = _place, _try
    try:
        yield
    finally:
        exec_e_run.Run._place, exec_e_run.Run._try_order = place, try_order


def run_point(sid: str) -> dict:
    inp = etap1.build_inputs(sid)
    evr = Evaluator(sid, inp, etap1.base_point(sid), 0.0004)
    pt = point(sid)
    orders: list = []
    with log_orders(orders):
        df, inv, eng, cnt, rs = evr.excess({**YAKOR, **pt}, "train")
    sk = tuple(sorted({k: pt[k] for k in evr.sk.sig_names}.items()))
    trades = []
    for run in rs:
        for tr in run.trades:
            m = inp.meta[tr.ticker]
            d = m["dates"]
            if not tr.reason or tr.entry_day < int(np.searchsorted(d, np.datetime64("2018-01-01"))):
                continue                                  # как в журнале: открытые на конце и прогрев-2017 не входят
            st = float(evr._sigs[sk][tr.ticker].strength[0][m["s0"] + tr.d_sig])
            pnl = tr.side * (tr.exit_px / tr.entry_px - 1.0) - tr.fee_in - tr.fee_out
            trades.append(dict(tiker=tr.ticker, d_in=str(d[tr.entry_day])[:10], d_out=str(d[tr.exit_day])[:10],
                               side=tr.side, pnl=float(f"{pnl:.6f}"), sila=st, d_sig=str(d[tr.d_sig])[:10]))
    for o in orders:
        o["d_sig"] = str(inp.meta[o["tiker"]]["dates"][o["d_sig"]])[:10]
    return dict(sid=sid, pt=pt, trades=trades, orders=orders, excess=df["excess"], vhod=pt["vhod"],
                vhody=inp.digest())


def journal(sid: str) -> list:
    rows = [r for r in csv.DictReader(l for l in open(ROOT / f"docs/tables/diag/{sid}_sdelki.csv", encoding="utf-8")
                                      if not l.startswith("#"))]
    return [dict(tiker=r["tiker"], d_in=r["data_vhoda"], d_out=r["data_vykhoda"], pnl=float(r["pnl"])) for r in rows]


if __name__ == "__main__":
    out = run_point(sys.argv[1])
    out["excess"] = out["excess"].to_dict()
    pickle.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else f"{sys.argv[1]}_zh066.pkl", "wb"))
    print(sys.argv[1], "сделок", len(out["trades"]), "заявок", len(out["orders"]))
