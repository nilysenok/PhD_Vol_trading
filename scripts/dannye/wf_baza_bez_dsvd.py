"""К-1 (σ̂ C/D на wf, шаги 9–10): базовая матрица признаков без выходных сессий MOEX (ДСВД).

Справка: docs/wf_sigma_910_dsvd.md, п. 3. Строки ≤ 2024 берутся из исходной матрицы без изменений. С 2025 строки дат ДСВД
(календарь docs/issledovaniya/2026-10-30_kalendar_moex_2025_2026.csv, tip = vyhodnaya_sessiya_DSVD) удаляются, собственные признаки
(OWN) пересчитываются m0_pit_lib.CutCalc (окно 09:50–18:40) на свечах без баров дат ДСВД: ночная доходность понедельника — от
закрытия пятницы, а не воскресенья. Лаги — add_har_lags/add_extended_lags, цели — m0_pit_lib.add_targets. Внешние столбцы не меняются.

Запуск: FASTBT_MAIN=<корень данных> python3 scripts/dannye/wf_baza_bez_dsvd.py --out <каталог> [--konec 2026-02-03]
"""
import argparse
import datetime as dt
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
KAL = REPO / "docs/issledovaniya/2026-10-30_kalendar_moex_2025_2026.csv"
MAIN = Path(os.environ.get("FASTBT_MAIN", "/Users/nilysenok/Desktop/Старое/PhD_Vol_trading"))
GRANICA = pd.Timestamp("2025-01-01")  # строки раньше — побайтно прежние
OKNO = (dt.time(9, 50), dt.time(18, 40))
TIP_DSVD = "vyhodnaya_sessiya_DSVD"
TARGETS = [f"rv_target_h{h}" for h in (1, 5, 22)] + [f"rv_target_mean_h{h}" for h in (1, 5, 22)]


def daty_dsvd(path=KAL):
    """Даты выходных сессий — только из календаря, не из дня недели."""
    k = pd.read_csv(path, parse_dates=["date"])
    return set(k.loc[k.tip == TIP_DSVD, "date"])


def _lib():
    sys.path.insert(0, str(REPO / "scripts/analysis"))
    import m0_pit_lib
    return m0_pit_lib


def svechi_bez_dsvd(candles_dir, dsvd, ot=GRANICA - pd.Timedelta(days=370)):
    """Класс CutCalc, у которого из свечей убраны бары дат ДСВД (и всё раньше `ot`: лагам достаточно ~370 дней)."""
    import polars as pl
    lib = _lib()
    ds = [d.date() for d in dsvd]

    class CutBezDsvd(lib.CutCalc):
        def __init__(self):
            super().__init__(*OKNO)
            self.candles_path = Path(candles_dir)

        def load_candles(self, ticker):
            df = super().load_candles(ticker)
            return df.filter((pl.col("begin").dt.date() >= ot.date()) & ~pl.col("begin").dt.date().is_in(ds))

    return CutBezDsvd()


def own_bez_dsvd(calc, tickers):
    """Собственные признаки калькулятором на свечах без ДСВД: OWN + n_bars по (date, ticker), лаги добавлены."""
    import polars as pl
    d = calc.calculate_all_tickers(sorted(tickers), verbose=False).to_pandas()
    d["date"] = pd.to_datetime(d["date"])
    p = calc.add_extended_lags(calc.add_har_lags(pl.from_pandas(d.sort_values(["ticker", "date"]))), columns=["jump"],
                               lags=[1, 5, 22]).to_pandas()
    p["date"] = pd.to_datetime(p["date"])
    return p


def sobrat(base, own, dsvd, konec, own_cols, add_targets=None):
    """Матрица без ДСВД. base — исходная матрица, own — признаки калькулятора, own_cols — OWN + n_bars."""
    b = base[(base.date <= konec) & ~base.date.isin(dsvd)].reset_index(drop=True)
    b["_i"] = np.arange(len(b))  # порядок строк исходной матрицы сохраняется
    old = b.date < GRANICA
    new = b[~old].drop(columns=list(own_cols)).merge(own[["date", "ticker", *own_cols]], on=["date", "ticker"], how="left")
    out = pd.concat([b[old], new[b.columns]], ignore_index=True).sort_values(["ticker", "date"], kind="stable")
    cols = [c for c in own_cols if c != "n_bars"]
    out[cols] = out.groupby("ticker")[cols].ffill()  # как assemble(fill='asof'): только прошлое, до первого наблюдения — NaN
    out = out.sort_values("_i").reset_index(drop=True)
    if add_targets is None:
        add_targets = _lib().add_targets
    t = add_targets(out[["date", "ticker", "rv"]].copy())
    t = t.set_index(["date", "ticker"])[TARGETS]
    k = pd.MultiIndex.from_frame(out[["date", "ticker"]])
    new_t = t.reindex(k).to_numpy()
    mask = (out.date >= GRANICA).to_numpy()
    out.loc[mask, TARGETS] = new_t[mask]  # цели ≤ 2024 остаются прежними (проверяется тестом и логом)
    return out.drop(columns="_i")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="каталог-корень выхода (кладёт data/regen/features_pit_v3.parquet)")
    ap.add_argument("--konec", default="2026-02-03")
    a = ap.parse_args()
    regen = MAIN / "data/regen"
    base = pd.read_parquet(regen / "features_pit_v3.parquet")
    lib = _lib()
    dsvd = daty_dsvd()
    calc = svechi_bez_dsvd(MAIN / "moex_discovery/data/dataset_final/01_stocks/candles_10m", dsvd)
    own = own_bez_dsvd(calc, base.ticker.unique())
    out = sobrat(base, own, dsvd, pd.Timestamp(a.konec), list(lib.OWN) + ["n_bars"])
    dst = Path(a.out) / "data/regen"
    dst.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst / "features_pit_v3.parquet", index=False)
    shutil.copy(regen / "features_pit_v3.config.json", dst / "features_pit_v3.config.json")
    print("записан", dst / "features_pit_v3.parquet", "строк", len(out), "последняя дата", out.date.max().date())


if __name__ == "__main__":
    main()
