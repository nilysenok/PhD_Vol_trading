"""G4 B2: правило дополнительной информации — порог 0,95 строго «>», только строки <= 2016, группа без столбцов, мутант «все строки»."""
import numpy as np
import pandas as pd

from models_v2.s5.b2 import info

IDX = pd.bdate_range("2015-01-01", "2018-12-31")


def _with_corr(x: np.ndarray, r: float, seed: int = 0) -> np.ndarray:
    e = np.random.default_rng(seed).standard_normal(len(x))
    xc = x - x.mean()
    e = e - e.mean()
    e = e - (e * xc).sum() / (xc * xc).sum() * xc    # строго ортогонален x (Accelerate: без @)
    return r * xc / xc.std() + np.sqrt(1 - r * r) * e / e.std()


def _base():
    x = np.random.default_rng(1).standard_normal(len(IDX))
    return pd.DataFrame({"a": x, "b": np.random.default_rng(2).standard_normal(len(IDX))}, index=IDX), x


def test_threshold_is_strict_and_excludes_only_above():
    base, x = _base()
    cand = pd.DataFrame({"lo": _with_corr(x, 0.94, 7), "hi": _with_corr(x, 0.96, 8), "dup": x * 3 + 1}, index=IDX)
    t = info.check(base, cand, {"G": ("lo", "hi", "dup")}).set_index("col")
    assert t.loc["lo", "status"] == "accepted" and t.loc["hi", "status"] == "excluded_corr" and t.loc["dup", "status"] == "excluded_corr"
    assert t.loc["hi", "with_col"] == "a"


def test_at_exactly_threshold_is_accepted(monkeypatch):
    base, x = _base()
    cand = pd.DataFrame({"c": _with_corr(x, 0.9)}, index=IDX)
    monkeypatch.setattr(info, "THRESH", abs(base["a"].corr(cand["c"])) - 0.0)   # |ρ| == порог -> не «>»
    m = pd.DatetimeIndex(base.index) <= pd.Timestamp(info.LAST)
    monkeypatch.setattr(info, "THRESH", abs(base["a"][m].corr(cand["c"][m])))
    assert info.check(base, cand, {"G": ("c",)})["status"].iloc[0] == "accepted"


def test_accepted_columns_join_pool_for_later_candidates():
    base, x = _base()
    y = np.random.default_rng(5).standard_normal(len(IDX))
    cand = pd.DataFrame({"n1": y, "n2": y * 2 - 1}, index=IDX)
    t = info.check(base, cand, {"G1": ("n1",), "G2": ("n2",)})
    assert list(t["status"]) == ["accepted", "excluded_corr"] and info.accepted(t) == {"G1": ("n1",), "G2": ()}


def test_only_rows_up_to_2016_used_and_mutant_all_rows_caught():
    base, x = _base()
    late = IDX > pd.Timestamp("2016-12-31")
    base["a"] = np.where(late, 30 * x, x)       # после 2016 масштаб ×30: дисперсия всей выборки определяется поздним хвостом
    c = _with_corr(x, 0.0)
    c[late] = 30 * x[late]                      # после 2016 — дубликат столбца a; до — независим
    cand = pd.DataFrame({"c": c}, index=IDX)
    assert info.check(base, cand, {"G": ("c",)})["status"].iloc[0] == "accepted"
    assert info.check(base, cand, {"G": ("c",)}, last="2018-12-31")["status"].iloc[0] == "excluded_corr"   # мутант ловится
    assert info.LAST == "2016-12-31"


def test_short_history_not_added():
    base, x = _base()
    c = pd.Series(np.nan, index=IDX)
    c.iloc[:50] = 1.0 + np.arange(50)
    t = info.check(base, pd.DataFrame({"c": c}), {"G": ("c",)})
    assert t["status"].iloc[0] == "excluded_n" and info.accepted(t)["G"] == ()
