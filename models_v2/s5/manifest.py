"""BASELINE_MANIFEST.json: точная спецификация Б-2 (fx_b2_g132_b) и HAR-J(б) (harj_b), sha кода, входов и прогнозов."""
from __future__ import annotations

import json

from models_v2.s4 import mkt, prov4
from models_v2.s5 import common, frame

CODE = ("models_v2/daily.py", "models_v2/har.py", "models_v2/metrics.py", "models_v2/prov.py", "models_v2/s2/features.py", "models_v2/s3/scale.py",
        "models_v2/s3/resid.py", "models_v2/s3/post.py", "models_v2/s4/design.py", "models_v2/s4/fit.py", "models_v2/s4/mkt.py", "models_v2/s4/rows.py",
        "models_v2/s4/runner.py", "models_v2/s4/select.py", "models_v2/run_fix.py")
FORECASTS = ("fix_forecasts_c.parquet", "forecasts_p1.parquet", "daily_p1.parquet", "stress_p3.parquet")
DEF = {
    "ln_rv": "ln RV_d (%²) минус ln RV_m", "ln_rv_w": "ln средней RV за 5 дней минус ln RV_m", "ln_rv_m": "ln RV_m (средняя за 22 дня) минус ln RV_m = 0 (константа в форме 'm')",
    "ln1p_j": "ln(1 + J_d/RV_m), J = max(RV−BV, 0) в %²", "ln_bv": "ln BV_d минус ln RV_m", "ln_rv_h1": "ln RV первого часа (бары до 10:50, пол 1e-6) минус ln RV_m",
    "ln_rv_h2": "ln RV последнего часа (бары с 17:50, пол 1e-6) минус ln RV_m", "rs_plus": "ln(max(RS+, 1e-6)) минус ln RV_m, RS+ — сумма r² баров с r>0",
    "rs_minus": "ln(max(RS−, 1e-6)) минус ln RV_m, RS− — сумма r² баров с r<0", "gap_rel": "(100·ln(open_1/close_пред))² / RV_m — ночной гэп",
    "dow": "день недели 0..4", "proxy_d": "ln RV равновзвешенного портфеля 17 бумаг (10-мин доходности, среднее по бумагам) минус ln RV_m",
    "proxy_w": "то же, среднее за 5 дней, минус ln RV_m", "proxy_m": "то же, среднее за 22 дня, минус ln RV_m",
    "avg_d": "среднее по 17 бумагам ln RV_d, минус ln RV_m", "avg_w": "то же по RV_w, минус ln RV_m", "avg_m": "то же по RV_m, минус ln RV_m",
    "gap_med": "медиана по 17 бумагам gap_rel", "jump_share": "доля 17 бумаг со скачком Хуанга–Таучена (z > Φ⁻¹(0,99)) в день d",
    "rel_d": "ln RV_d SBER − proxy_d (до вычитания ln S)", "rel_w": "ln RV_w SBER − proxy_w", "rel_m": "ln RV_m SBER − proxy_m",
    "imoex_d": "ln RV_d IMOEX (10-мин бары, ISS) минус ln RV_m SBER", "imoex_w": "то же по RV_w", "imoex_m": "то же по RV_m",
    "relx_d": "ln RV_d SBER − ln RV_d IMOEX", "relx_w": "ln RV_w SBER − ln RV_w IMOEX", "relx_m": "ln RV_m SBER − ln RV_m IMOEX"}


def spec() -> dict:
    ch = common.read_comment_csv(common.TAB / "fix_i2_choice_g132.csv")
    ch = ch[ch.chosen]
    hp = {int(r.Y): {"depth": int(r.depth), "min_child_weight": float(r.min_data), "reg_lambda": float(r.l2), "eta": float(r.lr),
                     "subsample": float(r.subsample), "colsample_bytree": float(r.colsample), "max_trees": int(r.max_trees)} for r in ch.itertuples()}
    return {
        "b2": {"name": common.B2, "forecast_file": "data/regen/v2_sber_h1/fix_forecasts_c.parquet",
               "target": "y_t = ln RV_{t+1} − ln RV_m,t (форма 'm', S = RV_m, 22 дня); σ̂² = exp(ln RV_m + f + s²/2)",
               "features": [{"name": n, "definition": DEF[n]} for n in frame.FEAT], "n_features": len(frame.FEAT),
               "library": "xgboost reg:squarederror, tree_method=hist, nthread=1, seed=42, base_score=взвешенное среднее цели",
               "hyperparameters_by_year": hp, "hp_selection": "LOYO по годам цели < 01.01.Y (y∈[2015,Y−1]), QLIKE(б), сетка 132 точек, выбор на конфигурации level_base (HP затем применены к форме 'm' с IMOEX)",
               "schedule": "переобучение ежедневно, окно расширяющееся (пары t+1 ≤ d, один сегмент, без склейки через остановку 2022)",
               "early_stopping": "ES-блок — последние 252 пары, эмбарго 1, терпение 100, потолок 3000 деревьев; финальная модель на всех парах окна с n* деревьями (И1)",
               "s2": "s² = среднее квадрата остатка предварительной модели на ES-блоке; в σ̂² входит как s²/2",
               "grid": "2271 дат d (общая для всех моделей), цель — следующий годный день", "mkt_columns": {"base": list(mkt.BASE), "imoex": list(mkt.IDX)}},
        "harj_b": {"name": common.HJ, "forecast_file": "data/regen/v2_sber_h1/forecasts_p1.parquet",
                   "target": "ln RV_{t+1}", "features": [{"name": n, "definition": "ln RV_d" if n == "ln_rv" else "ln средней RV за 5 дней" if n == "ln_rv_w" else
                                                          "ln средней RV за 22 дня" if n == "ln_rv_m" else "ln(1+J_d), J в %²"} for n in ("ln_rv", "ln_rv_w", "ln_rv_m", "ln1p_j")],
                   "method": "OLS с константой на расширяющемся окне (пары t+1 ≤ d, один сегмент); σ̂² = exp(f + s²/2), s² = SSR/(n−k)", "grid": "2271 дат"}}


def build(reproduction: dict | None, qlike: dict) -> dict:
    code = {f: common.sha_file(common.ROOT / f) for f in CODE}
    inputs = {"sber_bars": common.sha_file(common.ROOT / "data/iss_check/bars_10m_full/SBER.parquet"), "imoex_bars": common.sha_file(common.IMOEX),
              "rvi_1850": common.sha_file(common.ROOT / "data/fastbt/s9/rvi_1850.parquet"),
              "papers16": {n: common.sha_file(prov4.BARS_DIR / (n + ".parquet")) for n in prov4.PAPERS}}
    preds = {f: common.sha_file(common.OUT / f) for f in FORECASTS}
    return {**spec(), "qlike_recomputed": qlike, "sha_code": code, "sha_inputs": inputs, "sha_forecasts": preds,
            "reproduction": reproduction, "reproduced": None if reproduction is None else reproduction["reproduced"]}


def write(path, reproduction, qlike) -> dict:
    m = build(reproduction, qlike)
    path.write_text(json.dumps(m, ensure_ascii=False, indent=1))
    return m
