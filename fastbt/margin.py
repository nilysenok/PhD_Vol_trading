"""Сценарий с маржой (решение владельца 26.10, 00:55; отдельная таблица, в проверку гипотез не
входит): капитал в позиции = 20% номинала (гарантийное обеспечение), остальное — под ставку
денежного рынка (решения ЦБ); доходность позиций и комиссии те же. Сценарий срочного рынка,
данных фьючерсов нет. Плюс избыточный Шарп A и C по годам против средней ставки года.

python3 -m fastbt.margin --design fastbt/designs/final/main.yaml [--workers 10] [--boot-reps 1000]
(только --scope dev, 2018–2021; тестовые годы — fastbt.final_test)
Выход: docs/tables/tabl_4_marzha_2018_2021.csv, tabl_4_sharpe_vs_stavka_2018_2021.csv
(разделитель «;», десятичная запятая).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "docs" / "tables"
SCEN = {"спот: капитал = 100% номинала": 1.0, "маржа: капитал = 20% номинала": 0.2}
NOTE = "сценарий срочного рынка, данных фьючерсов нет"


def to_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep=";", decimal=",", index=False, float_format="%.4f")


def margin_table(pos, cfg, years, window, reps):
    from fastbt import msp
    from fastbt.final_test import GAMMAS, ce_boot
    from fastbt.report import boot_delta
    rate = msp.rate_cbr() if cfg.fix_rate else msp.rate_restored()
    s, e = window
    pos = pos[pos["test_year"].isin(list(years))]
    rows = []
    for scen, share in SCEN.items():
        ports = {ap: msp.portfolio(pos[pos["approach"] == ap], rate, cfg.msp_comm,
                                   cap_share=share).loc[s:e]
                 for ap in sorted(pos["approach"].unique())}
        for ap, p in ports.items():
            r = dict(scenario=scen, approach=ap, window=f"{s[:4]}–{e[:4]}",
                     sharpe_excess=msp.sharpe(p["excess"]),
                     excess_pct_year=float(p["excess"].mean() * 252 * 100),
                     total_pct_year=float(p["total"].mean() * 252 * 100),
                     capital_in_positions_pct=float(p["exposure"].mean() * share * 100),
                     note=NOTE if share < 1 else "")
            if ap != "A":
                j = p.index.intersection(ports["A"].index)
                d, pv, _, _ = boot_delta(p.loc[j, "excess"].values,
                                         ports["A"].loc[j, "excess"].values, reps)
                r.update(d_sharpe_vs_A=d, p_sharpe=pv)
                for g in GAMMAS:
                    r[f"d_ce_g{g}_pct"], r[f"p_ce_g{g}"], _, _ = ce_boot(
                        p.loc[j, "total"].values, ports["A"].loc[j, "total"].values, g, reps)
            rows.append(r)
    return pd.DataFrame(rows)


def sharpe_vs_rate(pos, cfg, years):
    from fastbt import msp
    rate = msp.rate_cbr() if cfg.fix_rate else msp.rate_restored()
    pos = pos[pos["test_year"].isin(list(years))]
    ports = {(ap, sc): msp.portfolio(pos[pos["approach"] == ap], rate, cfg.msp_comm, cap_share=sh)
             for ap in ("A", "C") for sc, sh in (("spot", 1.0), ("marzha20", 0.2))}
    rows = []
    for y in years:
        days = pd.date_range(f"{y}-01-01", f"{y}-12-31")
        kr = rate.reindex(rate.index.union(days)).sort_index().ffill().bfill().reindex(days)
        r = dict(year=y, key_rate_mean_pct=float(kr.mean() * 100))
        for (ap, sc), p in ports.items():
            ex = p.loc[p.index.year == y, "excess"]
            r[f"sharpe_excess_{ap}_{sc}"] = msp.sharpe(ex) if len(ex) > 2 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--design", required=True)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--boot-reps", type=int, default=1000)
    a = p.parse_args(argv)
    from fastbt.final_test import run_design
    pos, cfg, _ = run_design(a.design, "dev", 2017, None, a.workers, "margin_dev")
    yrs = list(cfg.years_bcd)
    t1 = margin_table(pos, cfg, yrs, ("2018-01-01", "2021-12-31"), a.boot_reps)
    t2 = sharpe_vs_rate(pos, cfg, yrs)
    TABLES.mkdir(parents=True, exist_ok=True)
    to_csv(t1, TABLES / "tabl_4_marzha_2018_2021.csv")
    to_csv(t2, TABLES / "tabl_4_sharpe_vs_stavka_2018_2021.csv")
    pd.set_option("display.width", 250)
    print(t1.round(3).to_string(index=False))
    print(t2.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
