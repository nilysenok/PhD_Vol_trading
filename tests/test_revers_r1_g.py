"""R-г: периоды карты таймфреймов — границы чтения и окно сигналов (синтетика, без данных)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import diag_timeframes as dt  # noqa: E402


def test_granicy_periodov():
    assert dt.PERIODS["test1"] == ("2022-01-01", "2025-01-01", "2021-01-01")
    assert dt.PERIODS["train4"][1] == "2020-01-01"


def test_okno_signalov_ne_vyhodit_za_period():
    dates = pd.bdate_range("2021-06-01", "2025-06-01")
    n = len(dates)
    c = 100 + np.cumsum(np.random.default_rng(0).normal(size=n))
    f = pd.DataFrame({"date": dates, "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1000})
    f["day_idx"] = np.arange(n)
    d, *_ = dt.one(f, c, "kontrtrend", "kazhdyj_bar", np.random.default_rng(1), pd.Timestamp("2022-01-01"),
                   pd.Timestamp("2025-01-01"), 3)[:1]
    assert len(d) > 0
    assert d.min() >= np.datetime64("2022-01-01") and d.max() < np.datetime64("2025-01-01")
