"""R-б (реверс раунд 1): отказ без поддержки периода, сверка ловит расхождение, плата за шорт считается от базы (синтетика)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "revers"))
import r1_b  # noqa: E402


def test_test1_bez_podderzhki_otkaz(monkeypatch):
    monkeypatch.setattr(r1_b.s9, "PERIODS_S9", {"train4": ((2016,), 2015)})
    with pytest.raises(SystemExit):
        r1_b.ustanovit_period("test1")


def test_metriki_plata_za_short_umenshaet_pnl():
    x, baza = np.full(10, 0.001), np.full(10, 0.0002)
    m0, m5 = r1_b.metriki(x, baza, 0.0), r1_b.metriki(x, baza, 0.05)
    assert m0["sum_pnl_pct"] == pytest.approx(1.0) and m5["sum_pnl_pct"] == pytest.approx(1.0 - 0.05 * 0.0002 * 10 * 100)


def test_sverka_lovit_raskhozhdenie():
    obr = pd.read_csv(r1_b.ROOT / "docs" / "tables" / "a_strogaya" / "razlozhenie.csv")
    df = obr.drop_duplicates(["portfel", "kod", "plata_zaem"]).copy()
    assert r1_b.sverka(df, 1e-9) == 0.0
    df["msr"] += 1e-3
    with pytest.raises(SystemExit):
        r1_b.sverka(df, 1e-9)
