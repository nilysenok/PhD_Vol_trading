"""A10 (Ж-206, волна A): PREDICTIONS.parquet по алгоритму — колонки, 2271 строка, QLIKE пересчитан, режим по R5 (PIT).
Мутации: режим по факту RV цели; QLIKE с σ вместо σ²; сумма n режимов != 2271."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models_v2 import metrics  # noqa: E402
from models_v2.s6 import diag, paths6  # noqa: E402

ALGS = list(diag.ALG)


def cols(a):
    return ["date", "RV_actual", a + "_forecast", "HAR_forecast", "QLIKE_" + a, "QLIKE_HAR", "regime", "features"]


def load(alg):
    d = paths6.REP / alg
    return pd.read_parquet(d / "PREDICTIONS.parquet"), paths6.read_csv(d / "ANNUAL_PERFORMANCE.csv"), paths6.read_csv(d / "REGIME_PERFORMANCE.csv")


def regime_ref(pr):
    ctl = paths6.controls()
    x = diag.inputs(pd.DataFrame({"d": ctl["d"]}))
    return diag.regime_pit(x["rv_w"].to_numpy(float), x["stress"].to_numpy())


@pytest.mark.parametrize("alg", ALGS)
def test_predictions_shape_and_qlike(alg):
    a = diag.ALG[alg][0]
    pr, an, rg = load(alg)
    assert list(pr.columns) == cols(a) and len(pr) == paths6.N_GRID
    ctl = paths6.controls()
    assert (pr["date"].to_numpy("datetime64[ns]") == ctl["target_date"].to_numpy("datetime64[ns]")).all()      # date — день цели
    assert np.abs(metrics.qlike(pr["RV_actual"], pr[a + "_forecast"]) - pr["QLIKE_" + a]).max() < 1e-12
    assert np.abs(metrics.qlike(pr["RV_actual"], pr["HAR_forecast"]) - pr["QLIKE_HAR"]).max() < 1e-12
    assert pr["features"].str.contains("28 признаков").all() and (pr[a + "_forecast"] > 0).all()
    assert diag.checks(alg, pr, an, rg) == []


@pytest.mark.parametrize("alg", ALGS)
def test_builder_reproduces_stored_predictions(alg):
    """Мутации в сборке (σ вместо σ², режим по факту) ловятся: сборка кодом = сохранённый файл побитно."""
    pd.testing.assert_frame_equal(diag.predictions(alg), load(alg)[0], check_exact=True)


@pytest.mark.parametrize("alg", ALGS)
def test_regime_is_pit_and_equals_recomputation(alg):
    pr, _, _ = load(alg)
    assert (pr["regime"].to_numpy() == regime_ref(pr)).all()


@pytest.mark.parametrize("alg", ALGS)
def test_mutation_regime_by_fact_rv_caught(alg):
    pr, _, _ = load(alg)
    fact = diag.regime_pit(pr["RV_actual"].to_numpy(float), pr["regime"].str.endswith("стресс").to_numpy())   # режим по факту цели
    assert (fact != regime_ref(pr)).mean() > 0.1


@pytest.mark.parametrize("alg", ALGS)
def test_mutation_qlike_with_sigma_instead_of_variance_caught(alg):
    a = diag.ALG[alg][0]
    pr, an, rg = load(alg)
    bad = pr.assign(**{"QLIKE_" + a: metrics.qlike(pr["RV_actual"], np.sqrt(pr[a + "_forecast"]))})
    assert diag.checks(alg, bad, an, rg) == ["годы != общий", "QLIKE пересчёт"] or "QLIKE пересчёт" in diag.checks(alg, bad, an, rg)


@pytest.mark.parametrize("alg", ALGS)
def test_mutation_regime_n_not_2271_caught(alg):
    pr, an, rg = load(alg)
    rg2 = rg.copy()
    rg2.loc[rg2.exclusive_set, "n"] -= 1
    assert "сумма n режимов" in diag.checks(alg, pr, an, rg2)


@pytest.mark.parametrize("alg", ALGS)
def test_annual_rows(alg):
    _, an, rg = load(alg)
    yrs = an[an.kind == "год"]
    assert list(yrs.slice) == ["год %d" % y for y in paths6.YEARS] and int(yrs.n.sum()) == paths6.N_GRID
    assert {"2017–2021", "2022", "2023–2025", "без 24.02.2022", "без топ-20 дней (по |ΔQLIKE к HAR|)"} <= set(an.slice)
    assert (rg[rg.label.isin(["низкая (нижний терциль)", "высокая (верхний терциль)"])].marking == "по факту, не PIT").all()
