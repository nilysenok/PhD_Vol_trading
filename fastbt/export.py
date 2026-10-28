"""Экспорт дневных рядов МСП в формате perstrat_daily_returns (для рисунков 4.x).

python3 -m fastbt.export <positions.parquet> --out data/fastbt/series/perstrat_daily_returns
затем: python3 scripts/analysis/figures_final.py --series <out> --out docs/tables
Пишет MSP_{A..D}_{inst,std}.csv, S{1..6}_{A..D}_{inst,std}.csv (date, trading_ret, total_ret,
excess_ret, exposure) и ../perstrat_results.csv, ../msp_results.csv (метрики, comm_label, comm).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from fastbt import msp

COMMS = {"inst": 0.0005, "std": 0.0006}
SNUM = {"S1_MeanRev": 1, "S2_Bollinger": 2, "S3_Donchian": 3, "S4_Supertrend": 4,
        "S5_PivotPoints": 5, "S6_VWAP": 6}


def _write(port, path):
    out = port[["r", "total", "excess", "exposure"]].copy()
    out.columns = ["trading_ret", "total_ret", "excess_ret", "exposure"]
    out.index.name = "date"
    out.to_csv(path)


def export(pos: pd.DataFrame, out: Path, rate: pd.Series, years=(2022, 2026)):
    out.mkdir(parents=True, exist_ok=True)
    pos = pos[pos["test_year"].between(*years)]
    res_msp, res_ps = [], []
    for ap in sorted(pos["approach"].unique()):
        pa = pos[pos["approach"] == ap]
        for lab, c in COMMS.items():
            port = msp.portfolio(pa, rate, c)
            _write(port, out / f"MSP_{ap}_{lab}.csv")
            res_msp.append(dict(approach=ap, comm_label=lab, comm=c, **msp.metrics(port)))
            for sname, i in SNUM.items():
                ps = pa[pa["strategy"] == sname]
                if ps.empty:
                    continue
                pp = msp.portfolio(ps, rate, c)
                _write(pp, out / f"S{i}_{ap}_{lab}.csv")
                res_ps.append(dict(strategy=f"S{i}", approach=ap, comm_label=lab, comm=c,
                                   **msp.metrics(pp)))
    pd.DataFrame(res_msp).to_csv(out.parent / "msp_results.csv", index=False)
    pd.DataFrame(res_ps).to_csv(out.parent / "perstrat_results.csv", index=False)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("positions")
    p.add_argument("--out", required=True)
    p.add_argument("--rate", choices=["cbr", "restored"], default="cbr")
    a = p.parse_args(argv)
    pos = pd.read_parquet(a.positions)
    pos["date"] = pd.to_datetime(pos["date"])
    rate = msp.rate_cbr() if a.rate == "cbr" else msp.rate_restored()
    export(pos, Path(a.out), rate)
    print(f"ряды → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
