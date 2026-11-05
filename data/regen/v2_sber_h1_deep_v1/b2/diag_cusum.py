"""B2: диагностика правила CUSUM на реальных барах — число срабатываний и доля дат, где окно сводится к нижней границе 252 пар."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models_v2 import har, prov  # noqa: E402
from models_v2.s2 import features as ft  # noqa: E402
from models_v2.s5.b2 import cusum, world  # noqa: E402


def main() -> None:
    F = ft.build(pd.read_parquet(prov.BARS))
    grid = har.eligible(har.Panel(F.dates, F.seg, F.rv, F.X[:, :4]))
    e = cusum.har_arrivals(F)
    rec = []
    for h in (5.0, 8.0):
        al = cusum.run(e, h)[2]
        r = np.array([cusum.restart_row(al, int(d)) for d in grid], dtype=object)
        age = np.array([int(d) - x if x is not None else -1 for d, x in zip(grid, r)])
        rec.append({"h": h, "alarms_total": int(al.sum()), "alarms_on_grid": int(al[grid].sum()), "share_dates_with_restart": float((age >= 0).mean()),
                    "share_window_is_floor_252": float(((age >= 0) & (age < 252)).mean()), "alarm_dates": ";".join(str(F.dates[i])[:10] for i in np.flatnonzero(al))})
    df = pd.DataFrame(rec)
    df.to_csv(world.OUT / "CUSUM_DIAG_B2.csv", index=False)
    print(df.drop(columns="alarm_dates").to_string())


if __name__ == "__main__":
    main()
