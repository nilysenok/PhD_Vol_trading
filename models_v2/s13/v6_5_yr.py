"""V6-5, G43: F_YEARS — годовые Δ QLIKE (F1 − CB1, F2 − CB1) по годам цели t⁺ 2017…2025 и 2026 (частичный), их СКО (ddof = 1, по годам) на 2017–2021 и 2017–2025 и число лет с Δ < 0 — главный вывод F (план V6, п. 7).
Δ — в единицах QLIKE и в % годового уровня K1 (тот же год). Чистая функция от прогнозов на общих датах."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_q as Q

YEARS = list(range(2017, 2027))
SUMMARY = (("2017_2021", 2017, 2021), ("2017_2025", 2017, 2025))
PAIRS = (("F1", "CB1"), ("F2", "CB1"))


def years_table(fs: dict, w: pd.DataFrame, pairs=PAIRS) -> pd.DataFrame:
    rv, tp = w["rv"].to_numpy(), w["t_plus"].to_numpy()
    yr = pd.DatetimeIndex(tp).year.to_numpy()
    a1 = Q.annual(Q.ql(rv, w["K1"].to_numpy()), tp, YEARS)
    rows = []
    for x, y in pairs:
        ax, ay = Q.annual(Q.ql(rv, fs[x]), tp, YEARS), Q.annual(Q.ql(rv, fs[y]), tp, YEARS)
        d = {k: ax[k] - ay[k] for k in ax}
        for k in d:
            rows.append(dict(pair="%s vs %s" % (x, y), section="год", window=str(k), n=int((yr == k).sum()), qlike_x=ax[k], qlike_y=ay[k], delta_qlike=d[k], delta_pct_K1=100 * d[k] / a1[k]))
        for name, lo, hi in SUMMARY:
            ks = [k for k in d if lo <= k <= hi]
            dq, dp = np.array([d[k] for k in ks]), np.array([100 * d[k] / a1[k] for k in ks])
            rows.append(dict(pair="%s vs %s" % (x, y), section="итог", window=name, n=int(len(ks)), sd_delta_qlike=float(dq.std(ddof=1)), sd_delta_pct_K1=float(dp.std(ddof=1)),
                             mean_delta_pct_K1=float(dp.mean()), years_neg=int((dq < 0).sum()), years_n=len(ks)))
    return pd.DataFrame(rows)
