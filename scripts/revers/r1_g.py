"""Реверс, раунд 1, лист R-г: карта таймфреймов 6 ТФ × 3 семейства (Ж-117). Движок — scripts/diag_timeframes.py (#477).

--period train4: 2016–2019, 100 перемешиваний — воспроизводит docs/tables/progon_e1/timeframes_golaya.csv (проверка, --out).
--period test1: 2022–2024 (прогрев 2021, чтение 10-минуток строго < 2025-01-01), 20 перемешиваний, справочно, без выбора.
Таблица: docs/tables/revers_r1/g_timeframes_<period>.csv; статус — в колонке status.
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import diag_timeframes as dt  # noqa: E402
from fastbt.data import TICKERS  # noqa: E402

CFG = {
    "train4": (100, 4, "разработка train4 2016-2019, до аудита; потолок, не результат"),
    "test1": (20, 3, "реверс раунд 1, 2022–2024, справочно, без выбора"),
    "v2": (100, 5, "разработка В2 2015–2019, справочно, без выбора"),
}
OUT_DIR = {"v2": ("progon_e1", "timeframes_v2.csv")}      # остальные — revers_r1/g_timeframes_<period>.csv


def run(period: str, out: Path) -> int:
    reps, years, status = CFG[period]
    with Pool(2) as pool:
        per = pool.map(dt.worker, [(k, t, period, reps) for k, t in enumerate(TICKERS)])
    res = [r for key in per[0] for r in dt.rows(key, [p[key] for p in per], years, status)]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text((shapka(period, status) if period in OUT_DIR else "")
                   + pd.DataFrame(res).round(4).to_csv(index=False), encoding="utf-8")
    return len(res)


def shapka(period: str, status: str) -> str:
    """Шапка для новых таблиц: статус, коммит кода, sha256 входов (10-минутки целиком, механически; читаются < конца периода)."""
    import hashlib
    import subprocess
    kom = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    end = dt.PERIODS[period][1]
    h = "".join(f"# вход: candles_10m/{t}.parquet sha256={hashlib.sha256((dt.MAIN / 'moex_discovery/data/raw/candles_10m' / f'{t}.parquet').read_bytes()).hexdigest()}\n"
                for t in TICKERS)
    return (f"# статус: {status}; период {dt.PERIODS[period][0]}…{end} (прогрев с {dt.PERIODS[period][2]}), чтение 10-минуток строго < {end}\n"
            f"# коммит кода: {kom}; 100 перемешиваний, seed 1000+k\n" + h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=sorted(CFG), required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    d, f = OUT_DIR.get(a.period, ("revers_r1", f"g_timeframes_{a.period}.csv"))
    out = a.out or ROOT / "docs" / "tables" / d / f
    print(f"{run(a.period, out)} строк -> {out}")


if __name__ == "__main__":
    main()
