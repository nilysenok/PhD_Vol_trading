"""Диагностика Ж-200 (а)/(б). Запуск: python3 models_v2/run_fix_diag.py   (процессов <= 4)"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from models_v2 import har, prov  # noqa: E402
from models_v2.s2 import features as ft  # noqa: E402
from models_v2.s4 import diag, prov4  # noqa: E402


def main():
    F = ft.build(pd.read_parquet(prov.BARS))
    grid = har.eligible(har.Panel(F.dates, F.seg, F.rv, F.X[:, :4]))
    print("сетка", len(grid), "последняя дата ряда", F.dates[-1])
    t0 = time.time()
    a, diff = diag.diag_a(F, grid)
    prov4.write_csv(a, prov4.TAB / "fix_diag_a.csv", "Д-а; сверка Б-2 с forecasts_pb: max|Δ ln σ̂²|=%.3g" % diff)
    print(a.to_string(), "\nсверка Б-2 с pb max|Δln|=", diff, "%.1f мин" % ((time.time() - t0) / 60))
    b = diag.diag_b(F, grid)
    prov4.write_csv(b, prov4.TAB / "fix_diag_b.csv", "Д-б; s² в выборке")
    print(b.to_string(), "%.1f мин" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
