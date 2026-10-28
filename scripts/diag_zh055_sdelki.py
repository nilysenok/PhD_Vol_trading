"""Ж-056: журнал сделок точки итерации 1 (последняя строка S{k}_itog_v2.csv) и варианта k_s = 2 для точек без стопа.
Диагностика, не выбор: train 2018–2019, без дивидендов; в ledger и реестр отбора НЕ пишется. Код движка не меняется:
Evaluator из etap1.fabrika-набора (build_inputs + Evaluator) -> excess() -> Run.trades.
Запуск: FASTBT_MAIN=$PWD python3 scripts/diag_zh055_sdelki.py S1"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastbt import etap1                                     # noqa: E402
from fastbt.etap1_eval import YAKOR, Evaluator               # noqa: E402

HEAD = "# диагностика, не выбор; train 2018–2019; без дивидендов (Ж-055/Ж-056); в ledger и реестр отбора не добавлено"
COLS = ["data_vhoda", "data_vykhoda", "tiker", "napravlenie", "tsena_vkhoda", "tsena_vykhoda", "prichina", "pnl", "god_vkhoda", "div_gap", "div_gap_dolya"]


def num(s):
    try:
        f = float(s)
        return int(f) if f == int(f) else f
    except ValueError:
        return s


def point(sid):
    f = ROOT / f"docs/tables/zh047/{sid}_itog_v2.csv"
    row = [r for r in csv.DictReader(l for l in open(f) if not l.startswith("#"))][-1]
    return {k[2:]: num(v) for k, v in row.items() if k.startswith("p_")}


def gaps():
    """{тикер: [(last_cum_date, dps)]} из docs/issledovaniya/2026-10-27_dividendy_2017_2021.csv (≤ 2019)."""
    out = {}
    for r in csv.DictReader(open(ROOT / "docs/issledovaniya/2026-10-27_dividendy_2017_2021.csv")):
        if r["last_cum_date"] and r["last_cum_date"] <= "2019-12-31":
            out.setdefault(r["ticker"], []).append((r["last_cum_date"], float(r["dps_rub"])))
    return out


def div_gap(tr, d, close, dv):
    """Гэп = первый торговый день после last_cum_date; в позиции, если вход ≤ cum-день < выход (удержание через ночь гэпа).
    Эффект по знаку позиции: −side·dps/close(cum); без дивидендов в PnL это недополученный/лишний доход."""
    eff, hit = 0.0, False
    for cum, dps in dv.get(tr.ticker, []):
        i = int(np.searchsorted(d, np.datetime64(cum), side="right")) - 1   # последний торговый день ≤ cum
        if i >= 0 and tr.entry_day <= i < tr.exit_day:
            hit, eff = True, eff - tr.side * dps / close[i]
    return hit, eff


def journal(evr, inp, pt):
    df, inv, eng, cnt, rs = evr.excess({**YAKOR, **pt}, "train")
    out, dv = [], gaps()
    for run in rs:
        for tr in run.trades:
            d = inp.meta[tr.ticker]["dates"]
            a = int(np.searchsorted(d, np.datetime64("2018-01-01")))
            if not tr.reason or tr.entry_day < a:           # открытые на конце и прогрев-2017 не входят
                continue
            hit, eff = div_gap(tr, d, inp.close_n6[tr.ticker], dv)
            pnl = tr.side * (tr.exit_px / tr.entry_px - 1.0) - tr.fee_in - tr.fee_out
            out.append([str(d[tr.entry_day])[:10], str(d[tr.exit_day])[:10], tr.ticker, "long" if tr.side > 0 else "short",
                        f"{tr.entry_px:.6g}", f"{tr.exit_px:.6g}", tr.reason, f"{pnl:.6f}", str(d[tr.entry_day])[:4],
                        "да" if hit else "нет", f"{eff:.6f}"])
    out.sort(key=lambda r: (r[0], r[2]))
    return out, df["excess"]


def write(path, rows, note):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        fh.write(HEAD + "\n" + note + "\n")
        w = csv.writer(fh)
        w.writerow(COLS)
        w.writerows(rows)


def main(sid):
    inp = etap1.build_inputs(sid)
    evr = Evaluator(sid, inp, etap1.base_point(sid), 0.0004)
    pt = point(sid)
    variants = [("", pt)] + ([("_ks2", {**pt, "k_s": 2.0})] if pt.get("k_s") == "net" else [])
    for suf, p in variants:
        rows, ex = journal(evr, inp, p)
        note = f"# точка: {p}; стороны-сделки: {len(rows)}"
        write(ROOT / f"docs/tables/diag/{sid}_sdelki{suf}.csv", rows, note)
        print(sid, suf or "base", "сделок", len(rows), "T", len(ex), "sum excess", float(ex.sum()))


if __name__ == "__main__":
    main(sys.argv[1])
