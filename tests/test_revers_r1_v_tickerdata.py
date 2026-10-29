"""r1_v_a_zashch читает open/close из настоящего TickerData (fastbt.signals.interface), а не из угаданных атрибутов o/c.
Падает на прежнем коде (`td.o` → AttributeError)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "revers"))
import r1_v_a_zashch as R  # noqa: E402
from fastbt.signals.interface import TickerData  # noqa: E402


def test_vnutridnevnaya_td_na_nastoyashchem_tickerdata():
    n = 4
    dates = np.arange("2022-01-03", "2022-01-07", dtype="datetime64[D]").astype("datetime64[ns]")
    o = np.array([100.0, 99.0, 101.0, 103.0])
    c = np.array([100.0, 102.0, 101.0, 105.0])
    z = np.ones(n)
    td = TickerData("AAA", dates, o, z, z, c, z, np.zeros(n))
    assert hasattr(td, "open") and not hasattr(td, "o")
    I = R.vnutridnevnaya_td(td)
    assert np.allclose(I, [np.log(102 / 99), np.log(101 / 101), np.log(105 / 103), 0.0])
