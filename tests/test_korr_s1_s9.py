"""Тесты scripts/analysis/korr_s1_s9.py (Ж-077): синтетика + воспроизведение закоммиченной таблицы."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("korr_s1_s9", ROOT / "scripts" / "analysis" / "korr_s1_s9.py")
K = importlib.util.module_from_spec(_spec)
sys.modules["korr_s1_s9"] = K
_spec.loader.exec_module(K)


def _synt(n: int = 200) -> pd.DataFrame:
    r = np.random.default_rng(1)
    a = r.normal(size=n)
    return pd.DataFrame({"x": a, "y": -a, "z": r.normal(size=n)})


def test_korrelyaciya_ryada_s_soboy_ravna_1():
    m = K.korr_matrix(_synt())
    assert np.allclose(np.diag(m.values), 1.0, atol=1e-12)


def test_znak_ne_pereputan():
    m = K.korr_matrix(_synt())
    assert m.loc["x", "y"] == pytest.approx(-1.0, abs=1e-12)  # мутация: перепутанный знак ряда ловится


def test_propusk_v_ryade_oshibka():
    d = _synt()
    d.iloc[3, 0] = np.nan
    with pytest.raises(ValueError):
        K.korr_matrix(d)


def test_tablica_vosproizvoditsya_pobaytno():
    df = K.load()
    assert df.index.max() <= pd.Timestamp("2019-12-31")
    assert K.to_csv_text(K.korr_matrix(df)) == K.CSV.read_text()
