"""V5-0, G4: цель RV = 1e4·Σ r² по M10-барам основной сессии; DCT-II (ортонормальное) сохраняет энергию; мутация «выброшен бар» ломает тождество."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.fft import dct

from models_v2 import daily as D
from models_v2.s12 import v5_0_bars as BA


def synth_bars(days=6, seed=0, drop=None):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range("2019-05-06", periods=days):
        t = d + pd.Timedelta(hours=10)
        px = 100.0
        for i in range(53):
            o = px
            px = o * float(np.exp(rng.normal(0, 0.002)))
            rows.append((t + pd.Timedelta(minutes=10 * i), o, px))
    b = pd.DataFrame(rows, columns=["begin", "open", "close"])
    return b.drop(index=drop).reset_index(drop=True) if drop is not None else b


@pytest.mark.parametrize("n", [30, 52, 53, 41])
def test_bands_partition_energy(n):
    x = np.random.default_rng(n).normal(size=n)
    s, lo, mi, hi = BA.band_energy(x)
    assert abs((lo + mi + hi) - s) / s <= 1e-12
    assert lo > 0 and hi >= 0


def test_unnormalized_dct_breaks_parseval():
    x = np.random.default_rng(1).normal(size=53)
    assert abs((dct(x, type=2) ** 2).sum() - (x * x).sum()) > 1.0        # без norm="ortho" энергия не сохраняется


def test_synthetic_bars_energy_equals_canonical_rv():
    b = synth_bars()
    t = BA.day_table(b)
    d, _ = D.build_daily(b, index_days=None)
    assert BA.rel(t.reindex(d.index)["rv_bars"], d["rv"]) <= 1e-10
    assert BA.rel(t[["e_low", "e_mid", "e_high"]].sum(axis=1).reindex(d.index), d["rv"]) <= 1e-10


def test_mutation_dropped_bar_breaks_identity():
    b = synth_bars()
    d, _ = D.build_daily(b, index_days=None)
    t = BA.day_table(synth_bars(drop=[7]))                  # выброшен 8-й бар первого дня
    day = d.index[0]
    assert abs(t.loc[day, "rv_bars"] - d.loc[day, "rv"]) / d.loc[day, "rv"] > 1e-6
    assert t.loc[day, "n"] == 52


def test_real_ticker_identity_all_days():
    b = BA.ticker_bands("AFLT").dropna(subset=["rv"])
    assert len(b) > 2500
    assert BA.rel(b["rv_bars"], b["rv"]) <= BA.TOL_ID
    assert BA.rel(b[["e_low", "e_mid", "e_high"]].sum(axis=1), b["rv_bars"]) <= BA.TOL_PARSEVAL
    assert b["n"].between(30, 53).all()
