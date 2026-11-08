"""V4-C, Z: Intraday Shape (Ж-221 п. 8) — доля RV дня d в последний час основной сессии и индекс концентрации Херфиндаля по 10-минутным квадратам доходностей дня d.
Последний час — часть CLOSE Ж-214 (`hv2_sched.part_by_clock`, бары с begin >= последний бар режима − 50 мин), уже посчитанная в `daily_<T>.parquet` как rv_close; Херфиндаль считается из баров основной
сессии тем же разбором доходностей (`hv2_m10.bar_returns`). Все значения дня d — из баров дня d; `lag` — только для теста-мутации «бары дня d+1» (боевое значение 0)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s9 import hv2_m10 as M


def herf_by_day(bars: pd.DataFrame, lag: int = 0) -> pd.Series:
    """Σ_j (q_j / Σq)² по барам дня, q = r² (доходность 10 мин); 1/n — равномерно, 1 — всё в одном баре. Индекс — дата дня."""
    r = M.bar_returns(bars)
    q = r["r"].to_numpy(float) ** 2
    g = pd.DataFrame({"date": r["date"].to_numpy(), "q": q, "q2": q * q}).groupby("date").sum()
    h = (g["q2"] / g["q"] ** 2).where(g["q"] > 0)
    h.name = "herf"
    return h.shift(-lag) if lag else h


def last_hour_share(daily: pd.DataFrame, lag: int = 0) -> pd.Series:
    """rv_close / rv дня (доля последнего часа); индекс — дата дня."""
    s = (daily["rv_close"] / daily["rv"]).where(daily["rv"] > 0)
    s.name = "lasth"
    return s.shift(-lag) if lag else s
