"""Таблицы Ж-200: метрики П4, критерий «исправлен» (К1, К2), DM к HAR-J(б). Критерий считается механически."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s3 import dm, pb

K1_RATIO = 0.95
PERIODS = {"2017_2021": (2017, 2021), "2023_2025": (2023, 2025)}
BASE = "harj_b"


def qlike_period(f: pd.DataFrame, lo: int, hi: int) -> float:
    y = f["target_date"].dt.year
    m = (y >= lo) & (y <= hi)
    return float(metrics.qlike(f["rv"][m], f["sigma2"][m]).mean())


RATIO_MAX = 1.3   # Ж-202: отношение MSE блока остановки к MSE обучения


def criterion(fc: pd.DataFrame, leak: pd.DataFrame, over: pd.DataFrame) -> pd.DataFrame:
    """fc — long-прогнозы (`<строка>_b` и HAR-J(б) = BASE); leak — work_daily и full по строкам; over — fix_overfit (year='all').
    Итог (Ж-202): Р (ratio_mean <= 1,3) и К2; К1 (full <= 0,95·work) — справочно."""
    ref = fc[fc.model == BASE]
    rq = {k: qlike_period(ref, *v) for k, v in PERIODS.items()}
    ratio = over[over.year.astype(str) == "all"].set_index("model")
    rows = []
    for r in leak.itertuples():
        f = fc[fc.model == r.model + "_b"]
        o = ratio.loc[r.model]
        row = {"model": r.model, "qlike_work": r.work_daily, "qlike_full": r.full, "ratio_full_work": r.full / r.work_daily,
               "mse_train": o.mse_train, "mse_es": o.mse_es, "ratio_es_train": o.ratio_mean}
        row["K1"] = bool(row["ratio_full_work"] <= K1_RATIO)
        row["R"] = bool(row["ratio_es_train"] <= RATIO_MAX)
        for k, v in PERIODS.items():
            row["qlike_" + k], row["harj_" + k] = qlike_period(f, *v), rq[k]
        row["K2"] = bool(all(row["qlike_" + k] <= rq[k] for k in PERIODS))
        row["fixed"] = row["R"] and row["K2"]
        rows.append(row)
    return pd.DataFrame(rows)


def metrics_all(fc: pd.DataFrame, rvi: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([metrics.metrics_table(fc, rvi), pb.extra_slices(fc)], ignore_index=True)


def dm_table(fc: pd.DataFrame, models: dict) -> pd.DataFrame:
    """models: {имя `<строка>_b`: причина}; выборки: весь период, 2017–2021, 2023–2025, 2022; Холм внутри выборки."""
    base = fc[fc.model == BASE].sort_values("d").reset_index(drop=True)
    y = base["target_date"].dt.year
    samples = {"all": y > 0, "2017_2021": y <= 2021, "2023_2025": (y >= 2023) & (y <= 2025), "2022": y == 2022}
    rows = []
    for s, m in samples.items():
        part = []
        for name, why in models.items():
            f = fc[fc.model == name].sort_values("d").reset_index(drop=True)
            assert np.array_equal(f["d"], base["d"])
            part.append({"sample": s, "model": name, "why": why,
                         **dm.dm(base["rv"][m], base["sigma2"][m], f["sigma2"][m])})
        adj = dm.holm(np.array([x["p"] for x in part]))
        rows += [{**x, "p_holm": a} for x, a in zip(part, adj)]
    return pd.DataFrame(rows)
