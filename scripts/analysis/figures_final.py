#!/usr/bin/env python3
"""Рисунки глав 3-4 из финальных рядов (старый σ_pred).

  python3 scripts/analysis/figures_final.py \
      --series moex_volatility/results/final/strategies/msp/perstrat_daily_returns --out figures/final

--series: папка дневных рядов МСП (MSP_{A..D}_{inst,std}.csv, S{1..6}_{A..D}_{inst,std}.csv); рядом
(на уровень выше) ожидаются perstrat_results.csv и msp_results.csv (для ставок комиссий и сверки MDD).
Для глав 3: --wf (прогнозы walk-forward *_annual.parquet) и --t2019 (прогнозы теста 2019).
Пишет fig_<гл>_<N>_<имя>.png + .csv, sverka.csv, sverka.md, status.csv.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402

import figures_final_ch3 as c3  # noqa: E402
import figures_final_ch4 as c4  # noqa: E402
import figures_final_ch4b as c4b  # noqa: E402
from figures_final_common import ROOT, VOL, Ctx  # noqa: E402

DEF_SERIES = VOL / "results/final/strategies/msp/perstrat_daily_returns"


def md_table(ctx):
    rows = ["| Рис. | Показатель | На рисунке | Эталон | Δ | Источник эталона | Флаг |", "|---|---|---|---|---|---|---|"]
    for c in ctx.checks:
        rows.append(f"| {c['fig']} | {c['item']} | {c['figure_value']:.4f} | {c['ref']:.4f} | {c['diff']:+.4f} | {c['src']} | "
                    f"{'РАСХОЖДЕНИЕ' if c['flag'] else 'ок'} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default=str(DEF_SERIES))
    ap.add_argument("--out", default=str(ROOT / "figures/final"))
    ap.add_argument("--wf", default=None)
    ap.add_argument("--t2019", default=None)
    ap.add_argument("--keyrate", default=None)
    ap.add_argument("--features", default=None)
    a = ap.parse_args()
    ctx = Ctx(a.series, a.out, a.wf, a.t2019, a.keyrate, a.features)
    steps = [c3.fig_3_5_to_3_8, c3.fig_3_9_3_11, c4.fig_4_0, c4.fig_4_1, c4.fig_4_2, c4.fig_4_3, c4.fig_4_4,
             c4.fig_4_5, c4.fig_4_6, c4.fig_4_7, c4.fig_4_8, c4b.fig_4_9, c4b.fig_4_10, c4b.fig_4_11_14,
             c4b.fig_4_12, c4b.fig_4_13, c4b.fig_4_15_16]
    for s in steps:
        s(ctx)
    pd.DataFrame(ctx.checks).to_csv(ctx.out / "sverka.csv", index=False)
    (ctx.out / "sverka.md").write_text(md_table(ctx) + "\n", encoding="utf-8")
    pd.DataFrame(ctx.status, columns=["figure", "status", "note"]).to_csv(ctx.out / "status.csv", index=False)
    bad = [c for c in ctx.checks if c["flag"]]
    print(f"\nсверка: {len(ctx.checks)} показателей, расхождений > допуска: {len(bad)}")
    for c in bad:
        print(f"  {c['fig']} {c['item']}: {c['figure_value']:.4f} против {c['ref']:.4f}")


if __name__ == "__main__":
    main()
