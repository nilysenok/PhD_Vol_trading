"""Глава 4, подход A — честный пересчёт (С2): маска [0], комиссия 0,05%, корпоративные
события, ставка ЦБ по решениям, Шарп mean(excess)/std(excess); МСП-A, бенчмарки, H3,
статистика отсева фазы 1. Тестовые годы A — 2020–2026 (как в методе).

python3 -m fastbt.chapter4_a [--workers 10] [--boot-reps 1000] [--smoke] [--leak-mask-old]

Выход (docs/tables/): tabl_4_msp_a.csv (окна), tabl_4_msp_a_po_godam.csv (годы 2020–2026),
tabl_4_9_h3.csv (H3), tabl_4_otsev_a.csv (отсев фазы 1); позиции —
<основная копия>/data/fastbt/a_honest.parquet.
--leak-mask-old — тот же A, но маска [1] (старая утечка: обучение до 1 янв Y+1); остальное
(комиссия, исправления, корпоративные события) не меняется. Пишет
tabl_4_msp_a_po_godam_mask1.csv и, если есть таблица маски [0], tabl_4_a_utechka.csv.

H3 (текст диссертации, введение, гипотеза H3; §4.4, табл. 4.9): МСП значимо превосходит
классические портфели различных пропорций акций и облигаций. Проверка здесь: для каждого
бенчмарка H0: Шарп(МСП-A) ≤ Шарп(бенчмарка); парный бутстрап по общим дням (seed 42),
p = доля Δ* ≤ 0; H3 по бенчмарку принимается при Δ > 0 и p < 0,05.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "docs" / "tables"
WINDOWS = {"2020–2026": ("2020-01-01", "2026-12-31"),
           "2022–2025": ("2022-01-01", "2025-12-31"),
           "2022–2026": ("2022-01-01", "2026-12-31")}
H3_WINDOWS = ("2022–2025", "2022–2026")   # индексы MCFTR/RGBITR в data/ — с 03.01.2022
ALPHA = 0.05


def otsev(metas: dict) -> pd.DataFrame:
    rows = []
    for tk, by_sid in metas.items():
        for sid, m in by_sid.items():
            for Y, inf in m["A_info"].items():
                rows.append(dict(ticker=tk, sid=sid, year=int(Y), **inf))
    df = pd.DataFrame(rows)
    cols = ["n_coarse", "n_sh_pos", "n_mdd_ok", "n_exp_ok", "n_pass"]
    g = df.groupby(["sid", "year"])[cols].sum().reset_index()
    tot = df.groupby("year")[cols].sum().reset_index().assign(sid="все")
    g = pd.concat([g, tot], ignore_index=True)
    g["pass_share"] = g["n_pass"] / g["n_coarse"]
    # единицы (тикер × стратегия), где никто не прошёл и взяты топ-10 по Шарпу
    z = df.assign(fallback=df["n_pass"] == 0).groupby(["sid", "year"])["fallback"].sum()
    zt = df.assign(fallback=df["n_pass"] == 0).groupby("year")["fallback"].sum()
    g["n_units_fallback"] = [int(z.get((s, y), 0)) if s != "все" else int(zt.get(y, 0))
                             for s, y in zip(g["sid"], g["year"])]
    return g


def by_year(port: pd.DataFrame, years) -> pd.DataFrame:
    from fastbt import msp
    rows = []
    for y in years:
        p = port.loc[port.index.year == y]
        if len(p) > 2:
            rows.append(dict(year=y, **msp.metrics(p)))
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--boot-reps", type=int, default=1000)
    p.add_argument("--smoke", action="store_true", help="1 тикер × 1 стратегия, без записи таблиц")
    p.add_argument("--leak-mask-old", action="store_true",
                   help="маска [1] (утечка теста), остальное как в честном пересчёте")
    a = p.parse_args(argv)
    from fastbt import bench, msp
    from fastbt.cache import MAIN
    from fastbt.config import Config
    from fastbt.data import TICKERS
    from fastbt.pipeline import run_all
    cfg = Config(approaches="A")
    if a.leak_mask_old:
        cfg = dataclasses.replace(cfg, mask_old=True)
    sfx = "_mask1" if a.leak_mask_old else ""
    tickers, sids = (["SBER"], ["S3"]) if a.smoke else (TICKERS, ["S1", "S2", "S3", "S4", "S5", "S6"])
    t0 = time.time()
    pos, metas = run_all(cfg, tickers, sids, a.workers, log=lambda m: None)
    t_run = time.time() - t0
    rate = msp.rate_cbr()
    pa = pos[pos["approach"] == "A"]
    port_all = msp.portfolio(pa, rate, cfg.msp_comm)                       # 2020–2026
    port22 = msp.portfolio(pa[pa["test_year"] >= 2022], rate, cfg.msp_comm)  # как печать
    rows = []
    for wname, (s, e) in WINDOWS.items():
        src = port_all if wname.startswith("2020") else port22
        m = msp.metrics(src.loc[s:e])
        m.update(window=wname, turnover_pct=float(src.loc[s:e, "turnover"].mean() * 100))
        rows.append(m)
    t_msp = pd.DataFrame(rows)
    t_yr = by_year(port_all, cfg.years_a)
    cmd = (f"python3 -m fastbt.chapter4_a --workers {a.workers} --boot-reps {a.boot_reps}"
           + (" --leak-mask-old" if a.leak_mask_old else ""))
    print(t_yr.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if a.leak_mask_old:
        if not a.smoke:
            TABLES.mkdir(parents=True, exist_ok=True)
            t_yr.to_csv(TABLES / f"tabl_4_msp_a_po_godam{sfx}.csv", index=False)
            t_msp.to_csv(TABLES / f"tabl_4_msp_a{sfx}.csv", index=False)
            base = TABLES / "tabl_4_msp_a_po_godam.csv"
            if base.exists():
                b = pd.read_csv(base).set_index("year")["sharpe"]
                m1 = t_yr.set_index("year")["sharpe"]
                ut = pd.DataFrame({"sharpe_mask0": b, "sharpe_mask1": m1})
                ut["delta_mask1_minus_mask0"] = ut["sharpe_mask1"] - ut["sharpe_mask0"]
                w0 = pd.read_csv(TABLES / "tabl_4_msp_a.csv").set_index("window")["sharpe"]
                w1 = t_msp.set_index("window")["sharpe"]
                wt = pd.DataFrame({"sharpe_mask0": w0, "sharpe_mask1": w1})
                wt["delta_mask1_minus_mask0"] = wt["sharpe_mask1"] - wt["sharpe_mask0"]
                ut = pd.concat([ut.reset_index().rename(columns={"year": "period"})
                                .astype({"period": str}),
                                wt.reset_index().rename(columns={"window": "period"})])
                ut.to_csv(TABLES / "tabl_4_a_utechka.csv", index=False)
                print(ut.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            pos.to_parquet(MAIN / "data" / "fastbt" / "a_mask1.parquet", index=False)
        print(f"позиции {t_run:.0f} с, всего {time.time() - t0:.0f} с")
        return 0
    t_h3 = pd.concat([bench.h3_table(port22, rate, a.boot_reps, *WINDOWS[w]).assign(window=w)
                      for w in H3_WINDOWS], ignore_index=True)
    t_h3["h3_accept"] = np.where(t_h3["p_boot"].isna(), "",
                                 np.where((t_h3["d_sharpe_msp_minus"] > 0)
                                          & (t_h3["p_boot"] < ALPHA), "да", "нет"))
    t_ot = otsev(metas)
    print(t_msp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(t_h3[["window", "portfolio", "sharpe", "d_sharpe_msp_minus", "p_boot", "h3_accept"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if not a.smoke:
        TABLES.mkdir(parents=True, exist_ok=True)
        for name, t in [("tabl_4_msp_a.csv", t_msp), ("tabl_4_msp_a_po_godam.csv", t_yr),
                        ("tabl_4_9_h3.csv", t_h3), ("tabl_4_otsev_a.csv", t_ot)]:
            t.to_csv(TABLES / name, index=False)
        (TABLES / "tabl_4_a_README.md").write_text(
            "# Таблицы главы 4, подход A (fastbt, маска [0])\n\n"
            "Каждая таблица ниже воспроизводится командой из корня репозитория:\n\n"
            f"- `tabl_4_msp_a.csv` — МСП-A по окнам: `{cmd}`\n"
            f"- `tabl_4_msp_a_po_godam.csv` — МСП-A по тестовым годам 2020–2026: `{cmd}`\n"
            f"- `tabl_4_9_h3.csv` — H3, бенчмарки MCFTR/RGBITR: `{cmd}`\n"
            f"- `tabl_4_otsev_a.csv` — отсев фазы 1: `{cmd}`\n"
            f"- `tabl_4_msp_a_po_godam_mask1.csv`, `tabl_4_msp_a_mask1.csv`, `tabl_4_a_utechka.csv`"
            f" — до/после утечки: сначала `{cmd}`, затем `{cmd} --leak-mask-old`\n\n"
            f"Конфигурация: `{json.dumps(cfg.to_dict(), default=str)}`\n\n"
            f"Время счёта позиций: {t_run:.0f} с.\n")
        out = MAIN / "data" / "fastbt" / "a_honest.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pos.to_parquet(out, index=False)
    print(f"позиции {t_run:.0f} с, всего {time.time() - t0:.0f} с")
    return 0


if __name__ == "__main__":
    sys.exit(main())
