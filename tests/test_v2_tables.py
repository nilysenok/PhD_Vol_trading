"""G2–G6: выходные файлы этапа 1 (после `python3 models_v2/run_p1.py`)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models_v2 import metrics, prov  # noqa: E402

OUT = prov.ROOT / "data/regen/v2_sber_h1"
TAB = prov.ROOT / "docs/tables/v2_sber_h1"
MODELS = {"rw", "rv_w", "rv_m", "har_a", "har_b", "harj_a", "harj_b"}


def csv(name):
    return pd.read_csv(TAB / name, comment="#")


@pytest.fixture(scope="module")
def fc():
    return pd.read_parquet(OUT / "forecasts_p1.parquet")


def test_headers_carry_current_code_sha():
    for n in ("p1_metrics.csv", "p1_coef.csv", "p1_arch_check.csv", "p1_leak.csv", "p1_excluded_days.csv"):
        h = (TAB / n).read_text(encoding="utf-8").splitlines()[0]
        assert h.startswith("# разработка; после аудита этапа 1 проверяющим; оценка 2017–02.2026 по Ж-197")
        assert prov.code_sha() in h, n
        assert prov.sha_file(prov.BARS)[:16] in h and prov.sha_file(prov.RVI)[:16] in h


def test_forecasts_one_grid_seven_models(fc):
    assert list(fc.columns) == ["d", "target_date", "model", "sigma2", "rv"] and set(fc.model) == MODELS
    grids = {tuple(g["d"]) for _, g in fc.groupby("model")}
    assert len(grids) == 1
    f = fc[fc.model == "rw"].sort_values("d")
    assert str(f["d"].iloc[0])[:10] == "2016-12-30" and str(f["target_date"].iloc[0])[:10] == "2017-01-03"
    assert str(f["target_date"].iloc[-1])[:10] == "2026-02-27"
    assert np.isfinite(fc["sigma2"]).all() and (fc["sigma2"] > 0).all() and (fc["rv"] > 0).all()


def test_no_forecast_across_stop_or_in_first_21_days_after(fc):
    f = fc[fc.model == "rw"]
    assert not ((f["d"] <= "2022-03-24") & (f["target_date"] >= "2022-03-25")).any()
    dly = pd.read_parquet(OUT / "daily_p1.parquet")
    pos = dly[dly.seg == 1].index
    assert f[(f["d"] > "2022-03-24")]["d"].min() == pos[21]


def test_target_is_next_valid_day(fc):
    dly = pd.read_parquet(OUT / "daily_p1.parquet")
    nxt = pd.Series(dly.index[1:], index=dly.index[:-1])
    f = fc[fc.model == "rw"]
    assert (f["target_date"].to_numpy() == nxt.reindex(f["d"]).to_numpy()).all()
    assert np.allclose(f["rv"].to_numpy(), dly["rv"].reindex(f["target_date"]).to_numpy(), rtol=0, atol=0)


def test_metrics_table_matches_independent_recompute(fc):
    m = csv("p1_metrics.csv")
    assert set(m.model) == MODELS and len(m[m["slice"] == "all"]) == 7
    for model, g in fc.groupby("model"):
        r = g["rv"] / g["sigma2"]
        row = m[(m.model == model) & (m["slice"] == "all")].iloc[0]
        assert np.isclose(row.qlike, (r - np.log(r) - 1).mean(), rtol=1e-12) and row.n == len(g)
        assert np.isclose(row.mean_r, r.mean(), rtol=1e-12)
        e = np.log(g["rv"] / g["sigma2"])
        assert np.isclose(row.rmse_ln, np.sqrt((e ** 2).mean()), rtol=1e-12)
        years = m[(m.model == model) & m["slice"].str.match(r"y20\d\d$")]
        assert years["n"].sum() == len(g)
    need = {"rvi_high", "rvi_low", "mar_jun_2020", "y2022", "y2017", "y2026"}
    assert need <= set(m["slice"])
    hl = m[(m.model == "rw") & m["slice"].isin(["rvi_high", "rvi_low"])]["n"].sum()
    assert 0 < hl < len(fc[fc.model == "rw"][fc.target_date < "2022-01-01"])  # первые 20 дат без режима


def test_rvi_regime_uses_only_past_dates():
    rvi = pd.DataFrame({"date": pd.bdate_range("2017-01-03", periods=30),
                        "close_RVI": [10.0] * 25 + [100.0] + [1.0] * 4})
    reg = metrics.rvi_regime(rvi["date"], rvi)
    assert reg.iloc[:20].isna().all() and reg.iloc[20] == "low"
    assert reg.iloc[25] == "high" and reg.iloc[26] == "low"  # медиана прошлого не видит d


def test_arch_check_ten_dates_under_1e8():
    a = csv("p1_arch_check.csv")
    assert len(a) == 10 and a["d_lnf"].max() < 1e-8 and a["d_coef_abs"].max() < 1e-8
    d = pd.to_datetime(a["d"])
    assert d.min() <= pd.Timestamp("2017-01-02") and d.max() >= pd.Timestamp("2026-02-01")


def test_coef_table_last_day_each_year(fc):
    c = csv("p1_coef.csv")
    assert set(c.model) == {"har", "harj"}
    d = pd.to_datetime(c[c.model == "harj"]["d"])
    assert [x.year for x in d] == list(range(2016, 2027))
    grid = pd.to_datetime(fc[fc.model == "rw"]["d"])
    for x in d:
        assert x == grid[grid.dt.year == x.year].max()
    assert c[c.model == "harj"][["b0", "b_d", "b_w", "b_m", "b_j", "s2"]].notna().all().all()
    assert c[c.model == "har"]["b_j"].isna().all()
