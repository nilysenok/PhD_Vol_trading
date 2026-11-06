"""BASELINE.json обоих алгоритмов: значения читаются из кода (fit.HP, fit._params, lgbfit.params, таблица I2), не набираются вручную."""
from __future__ import annotations

import json
from datetime import date

import lightgbm
import pandas as pd
import xgboost

from models_v2.s4 import fit
from models_v2.s5 import common
from models_v2.s5.d import lgbfit
from models_v2.s6 import diag, paths6

XGB_KEYS = ("objective", "max_depth", "min_child_weight", "eta", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "gamma", "seed")
LGB_KEYS = ("objective", "boosting_type", "num_leaves", "max_depth", "min_data_in_leaf", "learning_rate", "feature_fraction", "bagging_fraction", "bagging_freq",
            "lambda_l1", "lambda_l2", "min_gain_to_split", "seed")
LGB_DEFAULT = {"boosting_type": "gbdt (умолчание библиотеки)", "lambda_l1": 0.0, "min_gain_to_split": 0.0}
PILOT_REF = ("b2_m_lgbm_b__daily", "b2_252_lgbm_b__daily", "d_lgbm_f4", "obj_l3l")


def xgb_year(hp) -> dict:
    p = fit._params(hp, None, 0.0)
    return {**{k: p.get(k) for k in XGB_KEYS if k in p}, "gamma": 0.0, "reg_alpha": 0.0}


def lgb_year(hp) -> dict:
    p = lgbfit.params(hp, fit.SEED)
    return {**{k: p.get(k, LGB_DEFAULT.get(k)) for k in LGB_KEYS}, "num_threads": p["num_threads"], "deterministic": p["deterministic"]}


def per_year(i2: dict, f) -> dict:
    return {str(Y): {**f(hp), "depth_I2": hp.depth, "max_trees": hp.max_trees} for Y, hp in sorted(i2.items())}


def common_fields(alg: str, i2: dict) -> dict:
    pr = pd.read_parquet(paths6.REP / alg / "PREDICTIONS.parquet")
    a = diag.ALG[alg][0]
    q = float(pr["QLIKE_" + a].mean())
    return {"status": paths6.STATUS, "date": str(date.today()), "features": {"set": "Б-2 (m_mkx)", "n": diag.n_features(), "names": [c for c in pd.read_parquet(common.REP / "PREDICTIONS.parquet").columns if c.startswith("f_")]},
            "target": "ln RV_{t+1} − ln RV_m (форма m; прогноз σ̂² = exp(off + f + s²/2), s² — средний квадрат остатка на блоке остановки)",
            "window": "расширяющееся: все пары t+1 <= d", "retrain": "ежедневно (каждая дата сетки)", "early_stopping": "последние %d пар окна, эмбарго %d, терпение %d, потолок %d деревьев; финал на всех парах с k деревьями" % (fit.ES_N, fit.EMBARGO, fit.PATIENCE, fit.MAX_TREES),
            "seed": fit.SEED, "qlike_all": q, "qlike_recomputed_from": "metrics.qlike на %d датах" % len(pr), "n_dates": len(pr), "forecast_sha256": None, "trials_budget_used": 0}


def xgb_baseline(i2: dict) -> dict:
    st = json.loads((common.DEEP / "b2" / "st_BASE.json").read_text())
    b = common_fields("xgboost", i2)
    return {**b, "id": "fx_b2_g132_b (Б-2)", "algorithm": "XGBoost", "library": "xgboost %s (прогнозы получены ранее; версия при обучении не сохранена)" % xgboost.__version__,
            "origin": "fit_i1 (xgb.train, reg:squarederror) — подтверждено ast, см. INVENTORY", "forecast_path": "data/regen/v2_sber_h1/fix_forecasts_c.parquet (model=fx_b2_g132_b)",
            "forecast_sha256": paths6.sha_file(paths6.OLD_DATA[0] / "fix_forecasts_c.parquet"), "hp_source": "годовой LOYO по целям < 01.01.Y из сетки 132 (I2, s4/select.py)",
            "hyperparameters_by_year": per_year(i2, xgb_year), "n_estimators": {"kind": "ранняя остановка, число деревьев k на дату; среднее по датам из st_BASE.json", "mean": st.get("trees"),
            "note": "распределение k по датам не сохранено в прежних файлах (только среднее) — не подтверждено"}}


def lgb_baseline(i2: dict) -> dict:
    fcp = paths6.OUT / "lightgbm" / "fc_lgbm_b2.parquet"
    st = json.loads((paths6.OUT / "lightgbm" / "st_lgbm_b2.json").read_text())
    inv = paths6.read_csv(paths6.REP / "INVENTORY.csv").set_index("id")
    b = common_fields("lightgbm", i2)
    return {**b, "id": "lgbm_b2_analog", "algorithm": "LightGBM", "library": "lightgbm %s" % lightgbm.__version__, "forecast_path": paths6.rel(fcp), "forecast_sha256": paths6.sha_file(fcp),
            "origin": "LightGBM-аналог Б-2: тот же конвейер (engine_d.daily, lib=lgbm → lgbfit.fit_lgbx)", "hp_source": "НАСЛЕДОВАННЫЕ HP (I2 из сетки 132 XGBoost); собственного годового LOYO-выбора у LightGBM нет",
            "inherited_hp": True, "hyperparameters_by_year": per_year(i2, lgb_year),
            "n_estimators": {"kind": "ранняя остановка; среднее k по датам (st_lgbm_b2.json)", "mean": st["trees"]},
            "warning": "единственная строка без собственного подбора: сравнение «XGBoost лучше/хуже LightGBM» без этого раскрытия недопустимо",
            "pilot_reference_rows": {k: {"qlike": float(inv.loc[k, "qlike_recomputed"]), "role": "справочно, не baseline: другой конвейер/бюджет"} for k in PILOT_REF if k in inv.index}}


def stage(a) -> int:
    from models_v2 import run_fix
    i2 = run_fix.load_i2("i2_choice_g132")
    paths6.dump_json(xgb_baseline(i2), paths6.REP / "xgboost" / "BASELINE.json")
    paths6.dump_json(lgb_baseline(i2), paths6.REP / "lightgbm" / "BASELINE.json")
    for alg in ("xgboost", "lightgbm"):
        for n, obj in (("OPTUNA_TRIALS.csv", None), ("CHECKPOINT.json", None)):
            f = paths6.REP / alg / n
            if n.endswith("csv") and not f.exists():
                f.write_text("trial_id,wave,direction,config,metric_inner,seed,time\n")
            if n.endswith("json") and not f.exists():
                lim = {"xgboost": {"obj_feat": 45, "hp": 55}, "lightgbm": {"obj_feat": 45, "hp": 55}}[alg]
                paths6.dump_json({"algorithm": alg, "limits": {**lim, "temporal_regime": 60, "ens_hybrid": 40}, "used": {"obj_feat": 0, "hp": 0, "temporal_regime": 0, "ens_hybrid": 0}}, f)
    return 0
