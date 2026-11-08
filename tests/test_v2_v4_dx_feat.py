"""V4-DX (G26, пункт 7): квинтили признаков по прошлому окну (без будущего), известный ответ, флаги изломов, изоляция диагностики от путей прогноза."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s11 import v4_dx_feat as DF, v4_dx_iso as ISO


def frame(n_years=4, seed=0, per_year=300):
    rng = np.random.default_rng(seed)
    rows = []
    for y in range(2017, 2017 + n_years):
        d = pd.bdate_range("%d-01-02" % y, periods=per_year)
        rows.append(pd.DataFrame({"date": d, "year": y, "ell": rng.normal(0, 1, per_year)}))
    f = pd.concat(rows, ignore_index=True)
    f["d"] = 0.1 * f["ell"] + rng.normal(0, 0.01, len(f))
    return f


def test_first_year_has_no_quintiles_and_later_use_past_edges_only():
    f = frame()
    assert quintiles(f, 2017).isna().all()
    q = DF.quintile_labels(f, "ell", 2019)
    edges = f.loc[f["date"] < "2019-01-01", "ell"].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
    cur = f[f["year"] == 2019]["ell"].to_numpy()
    assert np.array_equal(q.to_numpy(), np.searchsorted(edges, cur, side="right") + 1.0)


def quintiles(f, y):
    return DF.quintile_labels(f, "ell", y)


def test_mutation_edges_from_future_data_are_caught():
    """Мутация: границы по всем данным (включая будущее) дают другие квинтили, чем по прошлому окну; изменение будущего не должно менять метки года."""
    f = frame()
    base = quintiles(f, 2018).to_numpy()
    g = f.copy()
    g.loc[g["year"] >= 2019, "ell"] *= 50.0
    assert np.array_equal(base, quintiles(g, 2018).to_numpy())            # будущее (2019+) не влияет на границы 2018
    whole = np.searchsorted(g["ell"].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy(), g[g["year"] == 2018]["ell"].to_numpy(), side="right") + 1.0
    assert not np.array_equal(base, whole)


def test_known_answer_conditional_mean_increases_with_quintile():
    f = frame(5)
    f["in_range"], f["qh"], f["qm"] = 0.0, 1.0, 1.0
    for b in DF.BREAKS:
        f["post_" + b] = 0.0
    monkey = f.copy()
    t = DF.features_table(monkey.assign(**{c: 0.0 for c in DF.FEATS if c != "ell"}))
    q = t[(t["slice"] == "2018_2021") & (t["feature"] == "ell") & t["quintile"].isin(["1", "2", "3", "4", "5"])].sort_values("quintile")
    assert len(q) == 5 and (np.diff(q["mean_d"].to_numpy()) > 0).all()
    a = t[(t["slice"] == "2018_2021") & (t["feature"] == "ell") & (t["quintile"] == "all")].iloc[0]
    assert a["spearman_d"] > 0.9
    assert not ((t["slice"] == "Y2017") & (t["feature"] == "ell")).any()


def test_break_flags_cover_28_days_from_fixed_dates():
    P = pd.DataFrame({"ticker": "AFLT", "date": pd.to_datetime(["2018-04-05", "2018-04-06", "2018-05-04", "2018-05-05"])})
    P["t_plus"], P["rv"], P["h"], P["k2"], P["k3"] = P["date"] + pd.Timedelta(days=1), 1.0, 1.0, 1.0, 1.0
    from models_v2.s11 import v4_io as IO
    P["ticker"] = IO.M16[0]
    Z = P[["ticker", "date"]].assign(**{c: 0.0 for c in DF.FEATS})
    f = DF.frame(P, Z)
    assert list(f["post_2018-04-06"]) == [0.0, 1.0, 1.0, 0.0]


def test_isolation_dx_not_imported_by_forecast_paths():
    assert ISO.violators() == []
    from models_v2.s11 import v4_isolation
    assert v4_isolation.violators() == []


@pytest.mark.parametrize("src", ["from models_v2.s11 import v4_dx_years\n", "import models_v2.s11.v4_dx_cold as c\n", "from . import v4_dx_feat\n", "from models_v2.s11.v4_dx_iso import violators\n"])
def test_mutation_dx_import_is_caught(src):
    assert ISO.imports_dx(src)


@pytest.mark.parametrize("src", ["from models_v2.s11 import v4_io\n", "x = 'v4_dx_years'\n", "# from models_v2.s11 import v4_dx_years\n"])
def test_clean_code_not_flagged(src):
    assert not ISO.imports_dx(src)


def test_mutation_forecast_path_file_importing_dx_is_listed():
    """Подложенный файл пути прогноза (в т. ч. v4_cs_*) с импортом v4_dx_* попадает в нарушители."""
    for name in ("v4_zz_mutant_forecast_path.py", "v4_cs_zz_mutant.py"):
        f = ISO.ROOT / "models_v2" / "s11" / name
        f.write_text("from models_v2.s11 import v4_dx_years\n", encoding="utf-8")
        try:
            assert "s11/" + name in ISO.violators()
        finally:
            f.unlink()
    assert ISO.violators() == []


def test_real_cs_and_forecast_modules_do_not_import_dx():
    root = ISO.ROOT / "models_v2" / "s11"
    for p in sorted(root.glob("v4_cs_*.py")):
        assert not ISO.imports_dx(p.read_text(encoding="utf-8")), p.name
