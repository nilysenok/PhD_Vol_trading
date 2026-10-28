"""К-23: пауза при скользящем Шарпе тени (контракт §2Б.5). Синтетика с известным ответом + мутации."""
import numpy as np
import pandas as pd
import pytest

from fastbt import e1_pauza as pz

X = [1, -1, 1, -3, -2, -4, 1, 2, 1, 1, 1.5, 0.5]
D = pd.bdate_range("2018-01-01", periods=len(X))
EXPECT = [False] * 4 + [True] * 5 + [False] * 3          # посчитано вручную, окно W=3, порог 0


def ten():
    return pd.Series(X, index=D, dtype=float)


def test_pauza_known_answer():
    assert pz.pauza_mask(ten(), 3, 0.0).tolist() == EXPECT


def test_sharpe_value_uses_ddof1_and_annual_scale():
    s = pz.rolling_sharpe(ten(), 3)
    w = np.array([1.0, -1.0, 1.0])                         # строки d−3…d−1 для d=3
    assert s.iloc[3] == pytest.approx(w.mean() / w.std(ddof=1) * np.sqrt(252))
    assert s.iloc[:3].isna().all()                         # неполное окно → NaN, пауза не включается


def test_window_excludes_current_row():
    a = pz.rolling_sharpe(ten(), 3)
    x2 = ten()
    x2.iloc[8:] = -100.0                                   # будущее/текущее не должно менять Шарп_8
    assert pz.rolling_sharpe(x2, 3).iloc[8] == pytest.approx(a.iloc[8])


def test_zero_std_and_empty_do_not_crash():
    flat = pd.Series(0.0, index=D)
    assert not pz.pauza_mask(flat, 3, 0.0).any()           # std = 0 → NaN → нет паузы
    assert pz.pauza_mask(pd.Series(dtype=float), 3, 0.0).empty


def test_allow_blocks_both_sides_and_p1_when_axis_missing():
    ctx = type("C", (), {"ten": {"A": ten()}})()
    lo, sr = pz.allow({"pauza_okno": 3, "pauza_porog": 0.0}, "A", D, ctx)
    assert (~lo).tolist() == EXPECT and (~sr).tolist() == EXPECT
    lo, sr = pz.allow({}, "A", D, ctx)
    assert lo.all() and sr.all()


def test_mutation_window_with_current_day_caught(monkeypatch):
    monkeypatch.setattr(pz, "ROW_LAG", -1)
    assert pz.pauza_mask(ten(), 3, 0.0).tolist() != EXPECT


def test_mutation_pause_lifted_same_day_caught(monkeypatch):
    """Строка d = +1 делает Шарп ≥ 0 в d=8 (окно 1, 2, 1); по контракту пауза держится до d=9."""
    assert pz.pauza_mask(ten(), 3, 0.0).iloc[8]
    monkeypatch.setattr(pz, "ROW_LAG", -1)
    assert not pz.pauza_mask(ten(), 3, 0.0).iloc[8]


def test_mutation_ddof0_caught(monkeypatch):
    base = pz.rolling_sharpe(ten(), 3)
    monkeypatch.setattr(pz, "DDOF", 0)
    assert not np.allclose(pz.rolling_sharpe(ten(), 3).dropna(), base.dropna())
