"""Глава 4, подход A — честный пересчёт (С2): маска [0], комиссия 0,05%, корпоративные
события, ставка ЦБ по решениям, Шарп mean(excess)/std(excess); МСП-A, бенчмарки, H3,
статистика отсева фазы 1.

python3 -m fastbt.chapter4_a [--workers 12] [--boot-reps 1000] [--smoke]
Выход: docs/tables/tabl_4_msp_a.csv, tabl_4_9_h3.csv, tabl_4_otsev_a.csv;
позиции — data/fastbt/a_honest.parquet.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "docs" / "tables"
WINDOWS = {"2022–2025": ("2022-01-01", "2025-12-31"), "2022–2026": ("2022-01-01", "2026-12-31")}


def otsev(metas: dict) -> pd.DataFrame:
    rows = []
    for tk, by_sid in metas.items():
        for sid, m in by_sid.items():
            for Y, inf in m["A_info"].items():
                rows.append(dict(ticker=tk, sid=sid, year=int(Y), **inf))
    df = pd.DataFrame(rows)
    g = df.groupby(["sid", "year"])[["n_coarse", "n_sh_pos", "n_mdd_ok", "n_exp_ok", "n_pass"]].sum()
    g["pass_share"] = g["n_pass"] / g["n_coarse"]
    return g.reset_index()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--boot-reps", type=int, default=1000)
    p.add_argument("--smoke", action="store_true", help="1 тикер × 1 стратегия, без записи таблиц")
    a = p.parse_args(argv)
    from fastbt import bench, msp
    from fastbt.config import Config
    from fastbt.data import TICKERS
    from fastbt.pipeline import run_all
    cfg = Config(approaches="A")
    tickers, sids = (["SBER"], ["S3"]) if a.smoke else (TICKERS, ["S1", "S2", "S3", "S4", "S5", "S6"])
    t0 = time.time()
    pos, metas = run_all(cfg, tickers, sids, a.workers, log=lambda m: None)
    t_run = time.time() - t0
    rate = msp.rate_cbr()
    pa = pos[(pos["approach"] == "A") & (pos["test_year"] >= 2022)]
    port = msp.portfolio(pa, rate, cfg.msp_comm)
    rows = []
    for wname, (s, e) in WINDOWS.items():
        m = msp.metrics(port.loc[s:e])
        m.update(window=wname)
        rows.append(m)
    t_msp = pd.DataFrame(rows)
    t_h3 = pd.concat([bench.h3_table(port, rate, a.boot_reps, s, e).assign(window=w)
                      for w, (s, e) in WINDOWS.items()], ignore_index=True)
    t_ot = otsev(metas)
    print(t_msp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(t_h3[["window", "portfolio", "sharpe", "d_sharpe_msp_minus", "p_boot"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if not a.smoke:
        TABLES.mkdir(parents=True, exist_ok=True)
        cmd = "python3 -m fastbt.chapter4_a --workers 12 --boot-reps " + str(a.boot_reps)
        for name, t in [("tabl_4_msp_a.csv", t_msp), ("tabl_4_9_h3.csv", t_h3),
                        ("tabl_4_otsev_a.csv", t_ot)]:
            t.to_csv(TABLES / name, index=False)
        (TABLES / "tabl_4_a_README.md").write_text(
            f"Воспроизведение: `{cmd}`\n\nКонфигурация: `{json.dumps(cfg.to_dict(), default=str)}`\n"
            f"Время счёта позиций: {t_run:.0f} с.\n")
        out = ROOT / "data" / "fastbt" / "a_honest.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pos.to_parquet(out, index=False)
    print(f"позиции {t_run:.0f} с, всего {time.time() - t0:.0f} с")
    return 0


if __name__ == "__main__":
    sys.exit(main())
