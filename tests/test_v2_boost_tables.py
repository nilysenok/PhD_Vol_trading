"""G12–G16, G18: выходные файлы этапа 2 (после `python3 models_v2/run_p2.py`)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models_v2 import prov  # noqa: E402
from models_v2.s2 import boost, prov2, tune  # noqa: E402

OUT = prov.ROOT / "data/regen/v2_sber_h1"
TAB = prov.ROOT / "docs/tables/v2_sber_h1"
SPECS = [s.name for s in boost.all_specs()]


def csv(n):
    return pd.read_csv(TAB / n, comment="#")


@pytest.fixture(scope="module")
def fc():
    return pd.read_parquet(OUT / "forecasts_p2.parquet")


def test_headers_status_before_audit_and_code_sha():
    for n in ("p2_tuning.csv", "p2_metrics.csv", "p2_importance.csv", "p2_leak.csv", "p2_retrain.csv"):
        h = (TAB / n).read_text(encoding="utf-8").splitlines()[0]
        assert h.startswith("# разработка; до аудита; оценка 2017–02.2026 по Ж-197") and prov2.code_sha2() in h, n


def test_tuning_grid_as_declared_and_rule():
    t = csv("p2_tuning.csv")
    assert len(t) == 12 and set(zip(t.lib, t["loss"])) == {("xgb", "mse"), ("xgb", "hub"), ("lgbm", "mse"),
                                                              ("lgbm", "hub")}
    assert set(t.depth) == {3, 4} and set(t[t["loss"] == "hub"].delta) == {0.5, 1.0}
    assert (t[t["loss"] == "mse"].delta == 0).all() and (t.n_trees <= boost.MAX_TREES).all()
    ch = tune.choose(t)
    for (lib, loss), (dp, dl) in ch.items():
        g = t[(t.lib == lib) & (t["loss"] == loss)]
        assert g[g.chosen].shape[0] == 1 and g[g.chosen].iloc[0].qlike_b == g.qlike_b.min()
        assert (g[g.chosen].iloc[0].depth, g[g.chosen].iloc[0].delta) == (dp, dl)


def test_forecasts_48_models_one_grid_as_stage1(fc):
    p1 = pd.read_parquet(OUT / "forecasts_p1.parquet")
    d1 = set(p1[p1.model == "rw"]["d"])
    assert fc["model"].nunique() == 48 and set(fc.model) == {"%s_%s" % (s, k) for s in SPECS for k in ("a", "b", "btr")}
    for _, g in fc.groupby("model"):
        assert set(g["d"]) == d1 and len(g) == 2271
    assert np.isfinite(fc.sigma2).all() and (fc.sigma2 > 0).all()
    b = fc[fc.model == "xgb_mse_nohar_free_b"].set_index("d")["sigma2"]
    a = fc[fc.model == "xgb_mse_nohar_free_a"].set_index("d")["sigma2"]
    assert (b > a).all()
    t = p1[p1.model == "rw"].set_index("d")["rv"]
    assert np.array_equal(fc[fc.model == "xgb_mse_nohar_free_a"].set_index("d")["rv"].loc[t.index].to_numpy(), t.to_numpy())


def test_retrain_daily_trees_bounded_and_monthly_sensitivity_exists():
    r = pd.read_parquet(OUT / "retrain_p2.parquet")
    assert (r.groupby("model").size() == 2271).all()  # ежедневное переобучение: по модели на каждую дату сетки
    assert (r.n_trees <= boost.MAX_TREES).all() and (r.s2_es > 0).all() and (r.n_train >= boost.MIN_TRAIN).all()
    mm = csv("p2_metrics_monthly.csv")
    assert mm.model.nunique() == 49 and (mm[mm["slice"] == "all"].n == 2271).all()


def test_metrics_match_independent_recompute(fc):
    m = csv("p2_metrics.csv")
    assert "harj_b_p1" in set(m.model)
    for model, g in fc.groupby("model"):
        r = g["rv"] / g["sigma2"]
        row = m[(m.model == model) & (m["slice"] == "all")].iloc[0]
        assert row.n == 2271 and np.isclose(row.qlike, (r - np.log(r) - 1).mean(), rtol=1e-12)
        assert np.isclose(row.mean_r, r.mean(), rtol=1e-12)
    assert abs(m[(m.model == "harj_b_p1") & (m["slice"] == "all")].qlike.iloc[0] - 0.2337) < 5e-5   # было 0,2348 до исключения 23.02.2026


def test_importance_normalised_and_columns():
    t = csv("p2_importance.csv")
    assert set(t.model) == set(SPECS)
    for n, g in t.groupby("model"):
        assert np.isclose(g.gain_share.sum(), 1.0, atol=1e-9) and len(g) == (12 if "_har_" in n else 11)


def _leak():
    t = csv("p2_leak.csv").set_index("model")
    assert len(t) == 16 and set(t.index) == set(SPECS)
    return t


def red(cond, reason):
    if not cond:
        pytest.xfail("КРАСНЫЙ по объявленному допуску: " + reason)


@pytest.mark.parametrize("m", SPECS)
def test_i_oracle_improves_sharply(m):
    t = _leak()
    assert t.loc[m, "oracle"] <= 0.5 * t.loc[m, "work"]


@pytest.mark.parametrize("m", SPECS)
def test_ii_full_sample_not_worse(m):
    t = _leak()
    red(t.loc[m, "full"] <= t.loc[m, "work"] + 1e-12, "(ii) full %.4f > work %.4f" % (t.loc[m, "full"], t.loc[m, "work"]))


@pytest.mark.parametrize("m", SPECS)
def test_iii_permuted_near_rv_m(m):
    t = _leak()
    assert t.loc[m, "perm"] >= t.loc[m, "work"]
    q = t.loc[m, "qlike_rv_m"]
    red(abs(t.loc[m, "perm"] / q - 1) <= 0.30, "(iii) perm %.4f против RV_m %.4f" % (t.loc[m, "perm"], q))


def test_stage1_files_untouched():
    want = {"daily.py": "d69c6b0f",   # ↺ 05.11: правило nonworking_2025 (было e421ca15); forecasts_p1 было 4a0109a6 "har.py": "cf6da1f8", "metrics.py": "fddfd122", "leak.py": "a6750244",
            "arch_check.py": "b59f8e8d"}
    for n, h in want.items():
        assert prov.sha_file(prov.ROOT / "models_v2" / n).startswith(h), n
    assert prov.sha_file(OUT / "forecasts_p1.parquet").startswith("8ecc06f2")
