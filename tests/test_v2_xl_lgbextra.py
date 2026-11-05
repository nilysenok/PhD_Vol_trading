"""A12 (Ж-206, волна A): LightGBM — S1/S2 (leaf-wise), искажение цели, статистика листьев, 5 seed.
Мутации: S1 без max_depth=-1; S2 с num_leaves=2^depth; clip99/drop5 не меняют цель; fit_var(base) != fit_lgbx; seed не влияет."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models_v2.s4 import fit  # noqa: E402
from models_v2.s5.d import lgbfit  # noqa: E402
from models_v2.s6 import engine6, lgb_extra, paths6  # noqa: E402

HP = fit.HP(depth=3, min_child=10.0, max_trees=60)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(5)
    n = 900
    X = rng.normal(size=(n, 6))
    y = 0.6 * X[:, 0] - 0.3 * X[:, 1] * X[:, 2] + 0.5 * rng.normal(size=n)
    y[::97] += 8.0                                              # экстремальные цели
    return X, y, np.arange(n - 1)


def test_var_params_variants():
    b, s1, s2 = (engine6.var_params(HP, 42, v) for v in ("base", "S1", "S2"))
    assert b["max_depth"] == 3 and b["num_leaves"] == 8
    assert s1["max_depth"] == -1 and s1["num_leaves"] == 8
    assert s2["max_depth"] == -1 and s2["num_leaves"] == 31
    assert {k: v for k, v in b.items() if k not in ("max_depth", "num_leaves")} == {k: v for k, v in s2.items() if k not in ("max_depth", "num_leaves")}
    with pytest.raises(ValueError):
        engine6.var_params(HP, 42, "S3")


def test_fit_var_base_equals_fit_lgbx_bitwise(data):
    X, y, idx = data
    a = engine6.fit_var(X, y, idx, HP, 42)
    b = lgbfit.fit_lgbx(X, y, idx, HP, None, None, 42)
    assert a["k"] == b["k"] and a["s2_es"] == b["s2_es"] and (a["predict"](X[-5:]) == b["predict"](X[-5:])).all()


def test_variants_and_ymods_change_forecast(data):
    X, y, idx = data
    base = engine6.fit_var(X, y, idx, HP, 42)["predict"](X[-20:])
    for v in ("S1", "S2"):
        assert not np.array_equal(engine6.fit_var(X, y, idx, HP, 42, v)["predict"](X[-20:]), base)       # мутация: S без эффекта
    for ym in ("clip99", "drop5"):
        assert not np.array_equal(engine6.fit_var(X, y, idx, HP, 42, "base", ym)["predict"](X[-20:]), base)
    assert not np.array_equal(engine6.fit_var(X, y, idx, HP, 43)["predict"](X[-20:]), base)              # seed действует


def test_apply_ymod_known_answer(data):
    _, y, idx = data
    c, i1 = engine6.apply_ymod(y, idx, "clip99")
    assert c.max() <= np.quantile(y[idx], 0.99) and (i1 == idx).all() and (c[y <= c.max()] == y[y <= c.max()]).all()
    _, i5 = engine6.apply_ymod(y, idx, "drop5")
    assert len(i5) == len(idx) - 5 and set(idx[np.argsort(y[idx])[-5:]]).isdisjoint(i5)
    assert (engine6.apply_ymod(y, idx, "none")[0] is y)


def test_leaf_stats_counts_known_structure(data):
    X, y, idx = data
    m = engine6.fit_var(X, y, idx, HP, 42)
    s = engine6.leaf_stats(m["fin"], 10)
    assert s["n_leaves"] >= m["k"] and s["leaf_min"] >= 1 and 0 <= s["share_lt_2min"] <= 1
    t = m["fin"].trees_to_dataframe()
    assert float(t[t.tree_index == 0].query("split_feature != split_feature")["count"].sum()) == pytest.approx(len(idx) * 0.8, rel=0.05)    # bagging LightGBM — выборка по Бернулли, не ровно 0,8


def test_files_seeds_and_variants():
    sd = paths6.read_csv(paths6.REP / "lightgbm/LGB_SEEDS.csv")
    assert list(sd.seed) == [42, 43, 44, 45, 46] and sd.qlike_all.between(0.21, 0.23).all()
    ex = paths6.read_csv(paths6.REP / "lightgbm/LGB_EXTRA.csv")
    v = ex[ex.kind.astype(str).str.startswith("leafwise_")].set_index("variant")
    assert list(v.index) == ["base", "S1", "S2"] and v.loc["S2", "qlike_all"] > v.loc["base", "qlike_all"]
    s = ex[ex.kind == "seeds_42_46"].set_index("metric").loc["qlike_all"]
    assert s["std"] == pytest.approx(sd.qlike_all.std(ddof=1), abs=1e-12) and s["min"] == sd.qlike_all.min()
    for tag in ("", "_s43", "_s46", "_S1", "_S2"):
        assert len(pd.read_parquet(paths6.OUT / "lightgbm" / ("fc_lgbm_b2%s.parquet" % tag))) == paths6.N_GRID


def test_lgb_leaves_fixed_dates_and_extremes():
    lv = paths6.read_csv(paths6.REP / "lightgbm/LGB_LEAVES.csv")
    assert list(pd.to_datetime(lv.d))[-1] == pd.Timestamp("2022-02-22") and len(lv) == 12 and (lv.share_lt_2min.between(0, 1)).all()
    ex = paths6.read_csv(paths6.REP / "lightgbm/LGB_EXTRA.csv")
    for k in ("extreme_clip99", "extreme_drop5"):
        e = ex[ex.kind == k]
        assert len(e) == 12 and e.ln_change.abs().max() > 0
    assert lgb_extra.fixed_dates.__doc__
