"""Шаг 0, train/validation: периоды не пересекаются; мутация «validation включает 2019» ловится (синтетика)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
import s9_shag0_train_val as T  # noqa: E402

IDX = pd.bdate_range("2018-01-01", "2021-12-31")


def test_periods_disjoint_and_cover():
    m = T.split_masks(IDX)
    a, b = m["train_2018_2019"], m["val_2020_2021"]
    assert not (a & b).any() and (a | b).all()
    assert IDX[a].year.max() == 2019 and IDX[b].year.min() == 2020


def test_mutation_validation_includes_2019(monkeypatch):
    monkeypatch.setitem(T.PERIODS, "val_2020_2021", (2019, 2021))
    with pytest.raises(AssertionError):
        T.split_masks(IDX)


def test_selftest_and_empty_validation():
    T.selftest()
    ex = np.random.default_rng(1).normal(0, 0.01, len(IDX))
    p = pd.DataFrame(dict(excess=ex, total=ex, exposure=0.0, turnover=0.0), index=IDX)
    pos = pd.DataFrame(dict(date=IDX, key="X|S1", pos=0.0))
    r = T.row_for("нулевая позиция", p, pos)
    assert r["sdelok_v_god_train_2018_2019"] == 0.0 and r["dni_v_rynke_pct_val_2020_2021"] == 0.0
