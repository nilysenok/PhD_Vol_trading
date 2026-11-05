"""B2: правило дополнительной информации (GATES_B2 §2): |Пирсон| > 0,95 с имеющимися — столбец не добавляется. Без цели; только строки <= 31.12.2016."""
from __future__ import annotations

import numpy as np
import pandas as pd

THRESH, MIN_N, LAST = 0.95, 100, "2016-12-31"


def check(base: pd.DataFrame, cand: pd.DataFrame, groups: dict, last: str = LAST) -> pd.DataFrame:
    """base — 28 столбцов Б-2, cand — кандидаты (индекс — даты, одинаковый); groups — {группа: (столбцы…)} в порядке обработки.
    -> строки (group, col, n, max_abs_corr, with_col, status in {accepted, excluded_corr, excluded_n})."""
    m = pd.DatetimeIndex(base.index) <= pd.Timestamp(last)
    pool = base[m].copy()
    rec = []
    for g, cols in groups.items():
        for c in cols:
            x = cand[c][m]
            n = int(x.notna().sum())
            if n < MIN_N:
                rec.append((g, c, n, np.nan, "", "excluded_n"))
                continue
            cors = pool.apply(lambda col: x.corr(col, min_periods=MIN_N)).abs()
            if cors.notna().any() and cors.max() > THRESH:
                rec.append((g, c, n, float(cors.max()), str(cors.idxmax()), "excluded_corr"))
                continue
            rec.append((g, c, n, float(cors.max()) if cors.notna().any() else np.nan, str(cors.idxmax()) if cors.notna().any() else "", "accepted"))
            pool[c] = x
    return pd.DataFrame(rec, columns=["group", "col", "n", "max_abs_corr", "with_col", "status"])


def accepted(tab: pd.DataFrame) -> dict:
    """{группа: (принятые столбцы)}; пустая группа — trial не тратится."""
    return {g: tuple(t[t.status == "accepted"]["col"]) for g, t in tab.groupby("group", sort=False)}
