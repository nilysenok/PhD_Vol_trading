"""Ранжирование дизайнов C и D по прогонам fastbt.dev (решение владельца 25.10, 19:20).

python3 -m fastbt.dev_rank [--filter v2h1] [--latest]
Читает <основная копия>/data/fastbt/dev_runs/*.json (их пишет каждый прогон fastbt.dev).
Правило: прошли (а) лучше A в каждом году, одно исключение; (б) худший год не хуже худшего A;
(в) избыточный Шарп за период выше, чем у A. Среди прошедших — сначала с положительным
избыточным Шарпом за период, затем максимальный средний ΔШарп X−A по годам.
Если ни один не положителен — первой строкой «ни один дизайн C/D не даёт положительного
избыточного Шарпа на <годы>».
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from fastbt.cache import MAIN

RUNS = MAIN / "data" / "fastbt" / "dev_runs"


def load(filt: str = "", latest: bool = True) -> pd.DataFrame:
    rows = []
    for p in sorted(RUNS.glob("*.json")):
        r = json.loads(p.read_text())
        if filt and filt not in r["design"] and filt not in (r.get("note") or ""):
            continue
        ev = r["eval"]
        for x in ("C", "D"):
            if x not in ev:
                continue
            e = ev[x]
            rows.append(dict(approach=x, design=r["design"].split("/")[-1].replace(".yaml", ""),
                             hash=r["hash"], time=r["time"], passed=e["passed"],
                             positive=e["positive"], period=e["period"],
                             period_A=ev["A_period"], mean_delta=e["mean_delta"],
                             win=f"{e['win']}/{e['n_years']}", worst=e["worst"],
                             worst_A=e["worst_A"], c_years=e["c_years"], c_worst=e["c_worst"],
                             c_period=e["c_period"],
                             years=" ".join(f"{v:.2f}" for v in e["years"].values())))
    df = pd.DataFrame(rows)
    if latest and len(df):
        df = df.sort_values("time").groupby(["approach", "design"], as_index=False).last()
    return df


def rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["approach", "passed", "positive", "mean_delta"],
                          ascending=[True, False, False, False])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--filter", default="")
    p.add_argument("--all", action="store_true", help="все прогоны, не только последний на дизайн")
    a = p.parse_args(argv)
    df = rank(load(a.filter, not a.all))
    if df.empty:
        print("нет прогонов")
        return 1
    pos = df[df["passed"] & df["positive"]]
    if pos.empty:
        print("ни один дизайн C/D не даёт положительного избыточного Шарпа на 2018–2021")
    pd.set_option("display.width", 250)
    cols = ["approach", "design", "passed", "positive", "period", "period_A", "mean_delta",
            "win", "worst", "worst_A", "years"]
    print(df[cols].round(2).to_string(index=False))
    for x in ("C", "D"):
        d = df[(df["approach"] == x) & df["passed"]]
        print(f"{x}: выбор — {d['design'].iloc[0] if len(d) else 'нет прошедших'}")
    out = MAIN / "data" / "fastbt" / f"dev_rank{('_' + a.filter) if a.filter else ''}.csv"
    df.to_csv(out, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
