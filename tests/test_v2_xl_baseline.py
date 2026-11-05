"""A6/A7 (Ж-206, волна A): BASELINE.json обоих алгоритмов — все поля, значения = код (fit._params, lgbfit.params, I2 по годам), не набраны вручную.
Мутация: подмена max_depth одного года в JSON -> сверка с кодом падает. LightGBM-аналог Б-2: 2271 строка, сетка Б-2, σ̂² > 0, пометка о наследованных HP."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models_v2 import run_fix  # noqa: E402
from models_v2.s6 import baseline, paths6  # noqa: E402

COMMON = ("features", "target", "window", "retrain", "early_stopping", "seed", "hyperparameters_by_year", "n_estimators", "status", "library", "forecast_sha256", "date")
XGB_FIELDS = ("objective", "max_depth", "min_child_weight", "eta", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "gamma", "seed")
LGB_FIELDS = ("objective", "boosting_type", "num_leaves", "max_depth", "min_data_in_leaf", "learning_rate", "feature_fraction", "bagging_fraction", "bagging_freq",
              "lambda_l1", "lambda_l2", "min_gain_to_split", "seed")


def load(alg):
    return json.loads((paths6.REP / alg / "BASELINE.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def i2():
    return run_fix.load_i2("i2_choice_g132")


def problems(b, i2, f, fields):
    """Поля на месте и значения по годам равны вычисленным из кода."""
    bad = ["нет поля " + k for k in COMMON if k not in b]
    exp = baseline.per_year(i2, f)
    got = b["hyperparameters_by_year"]
    bad += ["год %s нет" % y for y in exp if y not in got]
    for y, e in exp.items():
        for k in fields:
            if y in got and (k not in got[y] or got[y][k] != e[k]):
                bad.append("%s %s" % (y, k))
    return bad


def test_xgb_baseline_fields_equal_code(i2):
    b = load("xgboost")
    assert problems(b, i2, baseline.xgb_year, XGB_FIELDS) == []
    assert b["algorithm"] == "XGBoost" and b["trials_budget_used"] == 0 and "не подтверждено" in b["n_estimators"]["note"] and b["n_estimators"]["mean"] > 0
    assert 0.2204 < b["qlike_all"] < 0.2206                                   # Б-2: 0,2205 (R4)


def test_lgb_baseline_fields_equal_code_and_marked_inherited(i2):
    b = load("lightgbm")
    assert problems(b, i2, baseline.lgb_year, LGB_FIELDS) == []
    assert b["inherited_hp"] is True and "НАСЛЕДОВАННЫЕ" in b["hp_source"] and "недопустимо" in b["warning"]
    assert {"b2_m_lgbm_b__daily", "d_lgbm_f4"} <= set(b["pilot_reference_rows"]) and b["n_estimators"]["mean"] > 0
    hp = b["hyperparameters_by_year"]["2020"]
    assert hp["num_leaves"] == 2 ** hp["max_depth"] and hp["bagging_freq"] == 1 and b["hyperparameters_by_year"]["2020"]["boosting_type"].startswith("gbdt")


@pytest.mark.parametrize("alg,f,fields", [("xgboost", baseline.xgb_year, XGB_FIELDS), ("lightgbm", baseline.lgb_year, LGB_FIELDS)])
def test_mutation_max_depth_by_year_caught(i2, alg, f, fields):
    b = copy.deepcopy(load(alg))
    y = sorted(b["hyperparameters_by_year"])[3]
    b["hyperparameters_by_year"][y]["max_depth"] += 1
    assert problems(b, i2, f, fields) == ["%s max_depth" % y]
    del b["hyperparameters_by_year"][y]["max_depth"]
    assert problems(b, i2, f, fields) == ["%s max_depth" % y]
    del b["seed"]
    assert "нет поля seed" in problems(b, i2, f, fields)


def test_lgb_forecast_grid_and_sigma():
    fc = pd.read_parquet(paths6.OUT / "lightgbm" / "fc_lgbm_b2.parquet")
    ctl = paths6.controls()
    assert len(fc) == paths6.N_GRID == len(ctl)
    assert (fc["d"].to_numpy("datetime64[ns]") == ctl["d"].to_numpy("datetime64[ns]")).all()
    assert np.isfinite(fc["sigma2"]).all() and (fc["sigma2"] > 0).all()
    assert load("lightgbm")["forecast_sha256"] == paths6.sha_file(paths6.OUT / "lightgbm" / "fc_lgbm_b2.parquet")
    assert paths6.sigma_frame(fc).shape == (paths6.N_GRID, 4)
    with pytest.raises(ValueError):                                           # мутация: σ̂² <= 0 или обрезанная сетка
        paths6.sigma_frame(fc.assign(sigma2=-fc["sigma2"]))
    with pytest.raises(ValueError):
        paths6.sigma_frame(fc.iloc[:-1])


def test_optuna_trials_empty_and_checkpoint_zero():
    for alg in ("xgboost", "lightgbm"):
        lines = (paths6.REP / alg / "OPTUNA_TRIALS.csv").read_text().splitlines()
        assert len(lines) == 1 and lines[0].startswith("trial_id")
        assert not any(load_ck(alg)["used"].values())


def load_ck(alg):
    return json.loads((paths6.REP / alg / "CHECKPOINT.json").read_text())
