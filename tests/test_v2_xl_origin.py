"""A4 (Ж-206, волна A): происхождение алгоритма по коду (R1), не по названию. Мутации: в fitter подменён xgb.train -> lgb.train; класс по имени строки."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models_v2.s6 import origin, paths6  # noqa: E402

SRC_XGB = "def fit_lgbm_like(X, y):\n    return xgb.train({}, X)\n"          # обманчивое имя: lgbm в названии, в коде xgb.train
SRC_LGB = "def fit_xgb_like(X, y):\n    return lgb.train({}, X)\n"
SRC_MIX = "def mix(a, b):\n    return 0.5 * a + 0.5 * b\n"
SRC_OLS = "def ols(X, y):\n    return np.linalg.lstsq(X, y)[0]\n"


def by_name(src: str) -> str:  # ошибочный классификатор (мутация м2)
    return "LightGBM" if "lgb" in src.lower() else "XGBoost"


def test_synthetic_classes():
    assert origin.classify_src(SRC_XGB) == "XGBoost" and origin.classify_src(SRC_LGB) == "LightGBM"
    assert origin.classify_src(SRC_MIX) == "без бустера и OLS" and origin.classify_src(SRC_OLS) == "OLS"
    assert origin.classify_src(SRC_XGB + SRC_LGB) == "LightGBM+XGBoost"


def test_mutation_m1_xgb_train_replaced_by_lgb_train_changes_class():
    src, _ = origin.func_src(paths6.ROOT / "models_v2/s4/fit.py", "fit_i1")
    assert origin.classify_src(src) == "XGBoost"
    assert origin.classify_src(src.replace("xgb.train", "lgb.train")) == "LightGBM"


def test_mutation_m2_name_based_class_on_deceptive_row_caught():
    assert by_name(SRC_XGB) != origin.classify_src(SRC_XGB)                 # классификация по имени неверна
    assert by_name(SRC_LGB.replace("lgb.train", "xgb.train").replace("xgb_like", "lgb_like")) != origin.classify_src(SRC_LGB.replace("lgb.train", "xgb.train"))


@pytest.mark.parametrize("rid,expect", [("fx_b2_g132_b", "XGBoost"), ("b2_f4", "XGBoost"), ("c1_opt", "XGBoost"), ("d_lgbm_f4", "LightGBM")])
def test_real_rows_single_booster(rid, expect):
    df = paths6.read_csv(paths6.REP / "INVENTORY.csv").set_index("id")
    assert df.loc[rid, "model_class"] == expect and str(df.loc[rid, "verified"]).startswith("подтверждено")


@pytest.mark.parametrize("rid,parts", [("c2_h3_roll_f4", ("HAR-J(б) OLS", "XGBoost")), ("e3_roll", ("HAR-J(б) OLS", "LightGBM"))])
def test_real_rows_mixture(rid, parts):
    df = paths6.read_csv(paths6.REP / "INVENTORY.csv").set_index("id")
    cls = df.loc[rid, "model_class"]
    assert cls.startswith("смесь") and all(p in cls for p in parts)


def test_real_chains_by_code():
    assert origin.classify_chain(origin.CHAINS["fix"])["class"] == "XGBoost"
    assert origin.classify_chain(origin.CHAINS["d_lgb"])["class"] == "LightGBM"
    assert origin.classify_chain(origin.CHAINS["s6_lgb"])["class"] == "LightGBM"
    assert origin.classify_chain(origin.CHAINS["har"])["class"] == "OLS"
    assert origin.spec_name_uses_lib()
