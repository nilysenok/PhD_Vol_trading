"""Ворота: позиции fastbt (--leak-mask-old) против daily_positions_pre2026.parquet.

python3 -m fastbt.gate <positions.parquet> [--out docs/otchety/fastbt_gate.csv]
Доля совпадения позиций по (стратегия, подход, год), лишние/недостающие строки.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from fastbt.data import POS_PRE2026

KEY = ["date", "strategy", "ticker", "approach", "test_year"]


def compare(new: pd.DataFrame, tickers=None, strategies=None) -> pd.DataFrame:
    filt = [("tf", "=", "daily")]
    if tickers is not None:
        filt.append(("ticker", "in", list(tickers)))
    if strategies is not None:
        filt.append(("strategy", "in", list(strategies)))
    sav = pd.read_parquet(POS_PRE2026, filters=filt)
    sav["date"] = pd.to_datetime(sav["date"])
    new = new.copy()
    new["date"] = pd.to_datetime(new["date"])
    m = new.merge(sav, on=KEY, how="outer", suffixes=("_n", "_s"), indicator=True)
    m["eq"] = (m["position_n"] == m["position_s"]) & (m["_merge"] == "both")
    m["dr"] = (m["daily_gross_return_n"] - m["daily_gross_return_s"]).abs()
    g = m.groupby(["strategy", "approach", "test_year"])
    out = pd.DataFrame({
        "rows_saved": g["_merge"].apply(lambda s: int((s != "left_only").sum())),
        "rows_new": g["_merge"].apply(lambda s: int((s != "right_only").sum())),
        "only_new": g["_merge"].apply(lambda s: int((s == "left_only").sum())),
        "only_saved": g["_merge"].apply(lambda s: int((s == "right_only").sum())),
        "eq_share": g["eq"].mean(),
        "max_dret": g["dr"].max(),
    }).reset_index()
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("positions")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    new = pd.read_parquet(a.positions)
    res = compare(new, tickers=sorted(new["ticker"].unique()),
                  strategies=sorted(new["strategy"].unique()))
    if a.out:
        res.to_csv(a.out, index=False)
    tot_eq = (res["eq_share"] * res["rows_saved"]).sum() / max(res["rows_saved"].sum(), 1)
    print(f"совпадение позиций: {tot_eq:.6f}; групп с eq<1: {(res['eq_share'] < 1).sum()} "
          f"из {len(res)}; лишних строк {res['only_new'].sum()}, недостающих "
          f"{res['only_saved'].sum()}")
    bad = res[(res["eq_share"] < 1) | (res["only_new"] > 0) | (res["only_saved"] > 0)]
    if len(bad):
        print(bad.to_string(index=False, max_rows=40))
    return 0 if len(bad) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
