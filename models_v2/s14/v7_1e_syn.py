"""V7-1a-E: синтетические бары для тестов (53 слота в день, кластеризация волатильности). Данные SBER не читаются."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s14 import v7_0_seq as SQ
from models_v2.s14.v7_1e_pre import make_pack


def make_bars(n: int = 100, seed: int = 0, start: str = "2019-08-01") -> tuple:
    """-> (bars DataFrame, days). Лог-волатильность бара — AR(1) по барам (метка предобучения предсказуема)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n)
    rows, p, lv = [], 100.0, -6.0
    for d in days:
        for s in range(SQ.SLOTS):
            lv = -6.0 + 0.9 * (lv + 6.0) + 0.3 * rng.normal()
            o, p = p, p * np.exp(np.exp(lv) * rng.normal())
            v = int(rng.integers(100, 10000))
            rows.append((d + pd.Timedelta(minutes=600 + 10 * s), o, max(o, p) * 1.0005, min(o, p) * 0.9995, p, v, v * p))
    return pd.DataFrame(rows, columns=["begin", "open", "high", "low", "close", "volume", "value"]), days


def synth_pack(n: int = 100, W: int = 5, seed: int = 0, start: str = "2019-08-01") -> dict:
    bars, days = make_bars(n, seed, start)
    return make_pack(SQ.build_store(bars, days), W)
