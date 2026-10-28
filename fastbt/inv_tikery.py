"""Л-1 инвентаризации (Ж-073 п. 1): пересчёт S1–S6 по тикерам при комиссии 0,035 %. Точка — шапка
`docs/tables/diag/S{k}_sdelki_posle370.csv` (итерация 1), движок — штатный (`Evaluator.excess`), без счёта на сетках.
Диагностика: train 2018–2019; в ledger и реестр отбора ничего не пишется. Рядом — итог Sk при 0,04 % (сверка).
mSR тикера — тикер один на весь капитал (остальные строки обнулены, свободные деньги по ставке, как в портфеле).
Запуск: FASTBT_MAIN=$PWD FASTBT_BARS1M=<D25/bars1m> python3 -m fastbt.inv_tikery"""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import numpy as np

from fastbt.metriki_e import msr

ROOT = Path(__file__).resolve().parents[1]
DIAG, OUT = ROOT / "docs/tables/diag", ROOT / "docs/tables/inv"
COMM = 0.00035                      # 0,035 %
COMM_REF = 0.0004                   # 0,04 % — колонка для сверки
YEARS = (2018, 2019)
STATUS = "разработка 2018–2019, после аудита, потолок; диагностика, в ledger и реестр отбора не добавлено"
COLS = ["tiker", "mSR_2018", "mSR_2019", "sdelok_2018", "sdelok_2019", "vystavleno", "ispolneno", "dolya_ispolneniya",
        "pnl_sr_%", "dlit_bary_sr", "dlit_dni_sr", "dolya_long", "dolya_short", "pnl_long_sum_%", "pnl_short_sum_%",
        "pnl_sum_%", "ITOG_004_mSR_2018", "ITOG_004_mSR_2019", "ITOG_004_sdelok_2018", "ITOG_004_sdelok_2019",
        "ITOG_004_pnl_sum_%"]


def point_of(sid: str) -> dict:
    """Точка — шапка `# точка:` журнала posle370."""
    head = open(DIAG / f"{sid}_sdelki_posle370.csv", encoding="utf-8").readlines()[3]
    return ast.literal_eval(head.split("# точка: ", 1)[1].rsplit("; стороны-сделки", 1)[0])


def bars_between(cum, d0: int, b0: int, d1: int, b1: int) -> int:
    """Длительность в 5-мин барах: позиция бара = накопленное число баров до дня + номер бара в дне."""
    return int(cum[d1] + b1 - (cum[d0] + b0))


def trade_record(tr, cum, year: int, pnl: float) -> dict:
    return dict(side=tr.side, pnl=pnl, year=year, bary=bars_between(cum, tr.entry_day, tr.entry_bar, tr.exit_day, tr.exit_bar),
                dni=int(tr.exit_day - tr.entry_day))


def po_storonam(recs: list) -> tuple:
    """(сделок лонг, сделок шорт, Σ pnl лонг, Σ pnl шорт)."""
    lg, sh = [r for r in recs if r["side"] > 0], [r for r in recs if r["side"] < 0]
    return len(lg), len(sh), sum(r["pnl"] for r in lg), sum(r["pnl"] for r in sh)


def msr_years(ex) -> list:
    """mSR 2018 и 2019 по ряду избыточных дневных доходностей (pandas Series с датами)."""
    return [float(msr(ex[ex.index.year == y].to_numpy())) for y in YEARS]


def standalone_excess(eng, di, g, sides, expo, comm: float):
    """Избыточная доходность тикера, торгуемого один на весь капитал: те же g/sides/expo, без деления на cnt."""
    nd = len(eng.dates)
    r = np.bincount(di, g - sides * comm, nd)
    return eng._finish(r, np.bincount(di, g, nd), np.bincount(di, expo, nd), np.bincount(di, sides, nd), 1.0)["excess"]


def ticker_row(name: str, recs: list, ms: list, n_o: int, n_f: int) -> list:
    n = len(recs)
    nl, ns, pl, ps = po_storonam(recs)
    mean = lambda k: float(np.mean([r[k] for r in recs])) if n else float("nan")      # noqa: E731
    return [name, ms[0], ms[1], *[sum(r["year"] == y for r in recs) for y in YEARS], n_o, n_f,
            n_f / n_o if n_o else float("nan"), 100 * mean("pnl"), mean("bary"), mean("dni"),
            nl / n if n else float("nan"), ns / n if n else float("nan"), 100 * pl, 100 * ps, 100 * (pl + ps)]


def check_g2(rows: list, itog: list) -> None:
    """G2: Σ pnl по тикерам = итог Sk (1e-9); то же для числа сделок."""
    s, n = sum(r[15] for r in rows), sum(r[3] + r[4] for r in rows)
    if abs(s - itog[15]) > 1e-9 or n != itog[3] + itog[4]:
        raise SystemExit(f"G2: Σ pnl по тикерам {s} против итога {itog[15]}, сделок {n} против {itog[3] + itog[4]}")


def code() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def run_sk(sid: str, comm: float) -> dict:
    """Один прогон точки итерации 1 при комиссии comm: записи сделок и mSR по тикерам, итог Sk."""
    from fastbt import diag_zh069 as dz9
    from fastbt import etap1
    from fastbt.diag_posle370 import accrue_dividends
    from fastbt.diag_zh066_run import log_orders
    from fastbt.etap1_eval import YAKOR, Evaluator
    pt, inp = point_of(sid), etap1.build_inputs(sid)
    evr = Evaluator(sid, inp, etap1.base_point(sid), comm)
    orders: list = []
    with accrue_dividends(), log_orders(orders):
        df, inv, eng, _, rs = evr.excess({**YAKOR, **pt}, "train")
    off = np.cumsum([0] + [len(r.g) for r in rs])
    if off[-1] != len(inv):
        raise SystemExit(f"{sid}: строки тикеров {off[-1]} != {len(inv)}")
    out = {}
    for k, (t, r) in enumerate(zip(evr.sk.tickers, rs)):
        d = inp.meta[t]["dates"]
        cum = np.r_[0, np.cumsum([len(x.c) for x in inp.bars[t]])]
        s0 = int(np.searchsorted(d, np.datetime64("2018-01-01")))
        recs = [trade_record(tr, cum, int(str(d[tr.entry_day])[:4]), dz9.trade_pnl(tr))
                for tr in r.trades if tr.reason and tr.entry_day >= s0]
        sl = slice(off[k], off[k + 1])
        ex = standalone_excess(eng, inv[sl], r.g, r.sides, r.expo, comm)
        mine = [o for o in orders if o["tiker"] == t and o["d_sig"] >= s0]      # заявки прогрева 2017 не входят
        out[t] = dict(recs=recs, msr=msr_years(ex), n_o=len(mine), n_f=sum(o["isp"] for o in mine))
    ex = df["excess"]
    return dict(per=out, itog_msr=msr_years(ex), digest=inp.digest(), point=pt)


def table(res: dict, ref: dict) -> list:
    rows = [ticker_row(t, v["recs"], v["msr"], v["n_o"], v["n_f"]) for t, v in res["per"].items()]
    allr = [x for v in res["per"].values() for x in v["recs"]]
    itog = ticker_row("ITOG", allr, res["itog_msr"], sum(v["n_o"] for v in res["per"].values()),
                      sum(v["n_f"] for v in res["per"].values()))
    rr = [x for v in ref["per"].values() for x in v["recs"]]
    cmp = [*ref["itog_msr"], *[sum(x["year"] == y for x in rr) for y in YEARS], 100 * sum(x["pnl"] for x in rr)]
    check_g2(rows, itog)
    return rows + [itog + cmp]


def main() -> None:
    from fastbt.diag_zh066 import write_csv
    OUT.mkdir(parents=True, exist_ok=True)
    for sid in ["S1", "S2", "S3", "S4", "S5", "S6"]:
        res, ref = run_sk(sid, COMM), run_sk(sid, COMM_REF)
        rows = [r + [""] * (len(COLS) - len(r)) for r in table(res, ref)]
        meta = {"статус": STATUS, "коммит кода": code(), "хэш входов": res["digest"], "комиссия": "0,035 % (колонки ITOG_004 — 0,04 %)",
                "точка": str(res["point"])}
        write_csv(OUT / f"{sid}_po_tikeram_0035.csv", meta, COLS, rows)
        print(sid, rows[-1][:5], rows[-1][15], "| 0,04:", rows[-1][16:], flush=True)


if __name__ == "__main__":
    main()
