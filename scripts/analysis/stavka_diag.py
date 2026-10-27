"""Диагноз ставки ЦБ в разработке 2018–2021 (Ъ 27.10, 12:00): прежние ряды против ряда владельца.

    python3 scripts/analysis/stavka_diag.py          # печать; без аргументов
Читает только ряд ставки и даты торгов ≤ 2021 (бары не считает).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fastbt import msp  # noqa: E402

OWNER = ROOT / "fastbt" / "rates" / "key_rate_cbr_vladelec.csv"


def owner() -> pd.Series:
    d = pd.read_csv(OWNER, comment="#", parse_dates=["data_vstupleniya"])
    return d.set_index("data_vstupleniya")["stavka_pct"] / 100.0


def on_days(s: pd.Series, days) -> pd.Series:
    """Ряд на днях: последнее известное значение; до первого — первое (как bfill в msp.portfolio)."""
    return s.reindex(s.index.union(days)).sort_index().ffill().bfill().reindex(days)


def runs(x: pd.Series) -> pd.DataFrame:
    """Отрезки одного значения: начало, конец, значение, календарных дней."""
    g = (x != x.shift()).cumsum()
    r = x.groupby(g).agg(["first"]).rename(columns={"first": "value"})
    r["start"] = x.index.to_series().groupby(g.values).min().values
    r["end"] = x.index.to_series().groupby(g.values).max().values
    r["days"] = (r["end"] - r["start"]).dt.days + 1
    return r[["start", "end", "value", "days"]]


def diff_days(a: pd.Series, b: pd.Series, days) -> pd.DataFrame:
    x, y = on_days(a, days), on_days(b, days)
    m = (x - y).abs() > 1e-9
    return pd.DataFrame({"date": days[m], "a": x[m].values * 100, "b": y[m].values * 100})


def main() -> None:
    from fastbt import data
    days = pd.DatetimeIndex(sorted(data.daily_df()["date"].unique()))
    days = days[(days >= "2018-01-01") & (days <= "2021-12-31")]
    cal = pd.date_range("2018-01-01", "2021-12-31")
    o, cbr, rst = owner(), msp.rate_cbr(), msp.rate_restored()
    print("торговых дней 2018–2021:", len(days))
    print("rate_cbr (parquet) против владельца, торговые дни:\n", diff_days(cbr, o, days).to_string())
    print("то же, календарные дни:\n", diff_days(cbr, o, cal).to_string())
    print("rate_restored: первая дата", rst.index.min().date(), "значение", rst.iloc[0] * 100)
    print("rate_restored на торговых днях 2018–2021 (bfill):\n", runs(on_days(rst, days)).to_string())
    print("rate_cbr, отрезки 2018–2021:\n", runs(on_days(cbr, days)).to_string())
    d = pd.DatetimeIndex(sorted(set(o.index) | set(cbr[cbr.diff().fillna(1) != 0].index)))
    print("даты решений владельца / parquet:")
    for t in d:
        a, b = (o.get(t), cbr.get(t))
        if a is None or b is None or abs(a - b) > 1e-9:
            print(" ", t.date(), "владелец", None if a is None else a * 100, "parquet", None if b is None else b * 100)


if __name__ == "__main__":
    main()
