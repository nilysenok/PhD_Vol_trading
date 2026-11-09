"""V7-1a-M: таблица SBER для счёта (листы 1b+). Вызывается ТОЛЬКО из стадий 1b–4, не при импорте и не в `--check` стадии 1a. Только SBER, h = 1.
Колонки: H = K1, K2, K3 (замороженные входы листа 0, `v7_0_ctrl.aligned`), rv_next, t_plus, rv_d (RV дня d из `daily_SBER`), jshare = J/RV дня d (`daily_SBER`: j, rv). Ничего не пишет."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s14 import v7_0_ctrl as C, v7_0_io as IO


def load_tab() -> pd.DataFrame:
    w = C.aligned()
    d = pd.read_parquet(IO.DAILY)
    d.index = pd.DatetimeIndex(pd.to_datetime(d.index))
    rv_d = d["rv"].reindex(w.index)
    js = (d["j"] / d["rv"]).reindex(w.index)
    tab = pd.DataFrame({"H": w["K1"].to_numpy(float), "K2": w["K2"].to_numpy(float), "K3": w["K3"].to_numpy(float), "rv_next": w["rv"].to_numpy(float), "t_plus": w["t_plus"].to_numpy(), "rv_d": rv_d.to_numpy(float),
                        "jshare": js.to_numpy(float)}, index=w.index)
    if not (tab["t_plus"].is_monotonic_increasing and np.isfinite(tab[["H", "K2", "K3", "rv_next"]].to_numpy()).all()):
        raise ValueError("таблица SBER: t⁺ не возрастает или есть нечисловые значения")
    return tab
