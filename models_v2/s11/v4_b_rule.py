"""V4-B, G11: правило B5 (объявлено до счёта). B5 считается ТОЛЬКО если по OOF-прогнозам B2 на M16 × цели 2017–2021: c = mean(RV/F_B2) отличается от 1 (t Ньюи–Уэста, лаг 5, по датам, |t| ≥ 1,96)
и средние ln(RV/H) и ln(RV/M) одного знака. Иначе B5 не считается, конфигурация уходит в резерв."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s3 import dm as dmod
from models_v2.s11 import v4_b_ops as OP, v4_io as IO

T_CRIT = 1.96


def rule_stats(F2: pd.DataFrame) -> dict:
    """F2 — прогнозы B2 (ticker, date, t_plus, rv, h, k2, k3, F). Только M16 и цели 2017–2021."""
    y = F2["t_plus"].dt.year
    s = F2[F2["ticker"].isin(IO.M16) & (y >= 2017) & (y <= 2021)]
    x = (s["rv"] / s["F"]).groupby(s["date"]).mean().sort_index().to_numpy()      # панельное по дате отношение RV/F
    m = OP.mean_m(s["k2"], s["k3"])
    c, v = float(x.mean()), dmod.nw_variance(x, 5)
    t = (c - 1.0) / np.sqrt(v) if v > 0 else np.nan
    lh, lm = float(np.log(s["rv"] / s["h"]).mean()), float(np.log(s["rv"] / m).mean())
    return {"c": c, "t_nw5": float(t), "n_dates": len(x), "mean_log_rv_h": lh, "mean_log_rv_m": lm, "same_sign": bool(np.sign(lh) == np.sign(lm) and lh != 0.0), "c_differs": bool(abs(t) >= T_CRIT)}


def rule_passes(st: dict) -> bool:
    return bool(st["c_differs"] and st["same_sign"])


def cfgs_to_run(st: dict) -> list:
    """Конфигурации этапа: B1–B4, B6 всегда; B5 — только по правилу."""
    return ["B1", "B2", "B3", "B4", "B6"] + (["B5"] if rule_passes(st) else [])


def consistent(st: dict, computed) -> bool:
    """B5 посчитан тогда и только тогда, когда правило выполнено."""
    return ("B5" in set(computed)) == bool(st["c_differs"] and st["same_sign"])       # условие записано заново, не через rule_passes: подмена выбора не обходит проверку
