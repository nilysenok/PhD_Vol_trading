"""Лист R-в реверса: параметры периодов r1_v.py. Известный ответ: test1 — ровно 2022–2024, граница чтения 2025-01-01 (≥ 2025 не читается)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "revers"))
import r1_v as R  # noqa: E402


def test_test1_granitsy_ne_dalshe_2024():
    gody, start, t0, end, last = R.PERIODS["test1"]
    assert gody == (2022, 2023, 2024) and pd.Timestamp(end) == pd.Timestamp("2025-01-01")
    assert pd.Timestamp(last) < pd.Timestamp(end) and pd.Timestamp(start) < pd.Timestamp(t0)


def test_test1_bez_imoex_ostanavlivaetsya():
    with pytest.raises(SystemExit):
        R.nastroit("test1", None)


def test_sverka_lovit_raznitsu():
    et = pd.read_csv(R.ETALON)
    assert R.sverka(et)
    bad = et.copy()
    bad.loc[0, "sharpe"] += 0.5
    assert not R.sverka(bad)
