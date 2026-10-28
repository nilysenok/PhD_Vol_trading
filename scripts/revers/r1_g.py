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
}


def run(period: str, out: Path) -> int:
    reps, years, status = CFG[period]
    with Pool(2) as pool:
        per = pool.map(dt.worker, [(k, t, period, reps) for k, t in enumerate(TICKERS)])
    res = [r for key in per[0] for r in dt.rows(key, [p[key] for p in per], years, status)]
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res).round(4).to_csv(out, index=False)
    return len(res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=sorted(CFG), required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    out = a.out or ROOT / "docs" / "tables" / "revers_r1" / f"g_timeframes_{a.period}.csv"
    print(f"{run(a.period, out)} строк -> {out}")


if __name__ == "__main__":
    main()
