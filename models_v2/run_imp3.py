"""Важность признаков Г1 (= Б-1, xgb, MSE): доля gain остаточной модели f на конце каждого года сетки (П6). Модели не сохраняются.

Запуск: python3 models_v2/run_imp3.py   (1 процесс, десять подгонок).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models_v2 import har, prov  # noqa: E402
from models_v2.run_pb import TAB, params_from_p2  # noqa: E402
from models_v2.s2 import features as ft  # noqa: E402
from models_v2.s3 import gen, pb, prov3  # noqa: E402


def main():
    F = ft.build(pd.read_parquet(prov.BARS))
    grid = har.eligible(har.Panel(F.dates, F.seg, F.rv, F.X[:, :4]))
    D, _ = pb.build_designs(F)
    dp, dl = params_from_p2()[("xgb", "mse")]
    year = F.dates[grid].astype("datetime64[Y]").astype(int) + 1970
    last = {int(y): int(grid[year == y][-1]) for y in np.unique(year) if y >= 2017}
    rows = []
    for y, i in last.items():
        r = gen.run_design(D["b1"], pb.spec_of("xgb", False), dp, dl, i, [i])
        rows += [{"year_end": y, "feature": ft.NAMES[c], "gain_share": float(g), "trees": r["k"], "no_split": r["no_split"]}
                 for c, g in zip(D["b1"].cols, r["imp"])]
    prov3.write_csv(pd.DataFrame(rows), TAB / "p3_importance.csv", "Г1 = Б-1 xgb MSE, цель = остаток к HAR-J(б); gain нормирован; последняя дата сетки года")


if __name__ == "__main__":
    main()
