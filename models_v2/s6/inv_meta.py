"""Метаданные семейств строк INVENTORY (п. 1 задания). Что не разобрано по коду — «не подтверждено» с причиной (R1)."""
from __future__ import annotations

import lightgbm
import xgboost

from models_v2.s6 import origin

UNK = "не подтверждено: поле не разобрано по коду в волне A"
FIXED = "lr 0,03; min_leaf 50; l2 5; subsample 0,8 (s2/boost.py FIXED); глубина и δ — s2/tune.py (Ж-197, цели 2016); seed 42"
LIBV = {"XGBoost": "xgboost %s (версия при обучении не сохранена)" % xgboost.__version__,
        "LightGBM": "lightgbm %s (версия при обучении не сохранена)" % lightgbm.__version__}
BASE = dict(objective=UNK, target=UNK, features=UNK, hp=UNK, window=UNK, retrain=UNK, es=UNK, seed=UNK, chain_key=None)


def fam(chain_key, **kw) -> dict:
    return {**BASE, "chain_key": chain_key, **kw}


I1 = "ранняя остановка по последним 252 парам, эмбарго 1, терпение 100, потолок 3000 (s4/fit.py)"
B2T = "ln RV_{t+1} − ln RV_m (форма m, s4/design.py)"


def pilot(name: str, src: str) -> dict:
    """src — имя файла прогнозов без расширения. Пилот p2/pb: библиотека — по полю Spec.lib (origin.spec_name_uses_lib)."""
    if src == "forecasts_p1":
        k = "naive" if name in ("rv_m", "rv_w", "rw") else "har"
        return fam(k, objective="OLS по ln RV_{t+1}" if k == "har" else "наивный прогноз", target="ln RV_{t+1}", seed="—", es="—", retrain="по дням (har.py)")
    if src == "forecasts_p3":
        return fam("naive", objective="постобработка: g1 — σ̂²·c над pb XGBoost; g2…g5 — HAR-J(б) и xgb_mse_nohar_mono_b (run_p3.py, строки 18–20)",
                   target="смесь/масштаб над прогнозами", es="—", seed="—")
    if src.startswith("fix_forecasts"):
        return fam("fix", objective="reg:squarederror", target=B2T if "b2" in name or "b1" in name else UNK, es=I1, seed="42", retrain="ежедневно",
                   window="расширяющееся (пары t+1<=T)", hp="годовой LOYO-выбор из сетки 132 (I2), s4/select.py" if "g132" in name else UNK)
    lib = "lgbm" if name.split("_")[0] == "lgbm" or "_lgbm" in name else "xgb"
    key = "b3" if name.startswith("b3_") else "p2_" + lib
    obj = {"xgb": "reg:squarederror" if "hub" not in name else "reg:pseudohubererror", "lgbm": "regression" if "hub" not in name else "huber"}[lib]
    return fam(key, objective=obj, target="ln RV_{t+1} (s2/features.py)", features="12 базовых" + (" + HAR-признак" if "_har" in name and "nohar" not in name else ""),
               hp=FIXED, window="расширяющееся", retrain="ежедневно (…__daily) / ежемесячно (…__monthly)" if src.endswith("pb") else UNK,
               es="последние 252 пары, эмбарго 1 (s2/boost.py split_pairs)", seed="42")


def table_row(rid: str, sheet: str) -> dict:
    """B1/B2: строки таблиц; C/D — отдельно в inv.py."""
    if sheet == "B1":
        if rid == "obj_l3l":
            return fam("b1_obj_lgb", objective="gamma (LightGBM)", target="уровень RV", hp="HP I2 (сетка 132)", es=I1)
        if rid.startswith("obj_"):
            return fam("b1_obj_xgb", objective=rid, target="см. описание строки в OBJECTIVE_EXPERIMENTS", hp="HP I2", es=I1, seed="42")
        if rid.startswith("ctrl_"):
            return fam("fix" if rid != "ctrl_harj" else "har", objective="контроль", target=B2T, es=I1, seed="42")
        return fam("fix", objective="смесь/остаток к HAR-J(б) или режим над XGBoost Б-2 (см. rowsfc.py)", target=UNK, es=I1, seed="42")
    return fam("b2_xgb", objective="reg:squarederror", target=B2T, es=I1, seed="42", hp="HP I2 (сетка 132)", features="Б-2 + группа/окно по строке (s5/b2/engine.py)")


def klass(m: dict) -> dict:
    """Класс и ссылки по цепочке функций обучения (ast)."""
    c = origin.classify_chain(origin.CHAINS[m["chain_key"]]) if m["chain_key"] else {"class": UNK, "refs": "", "chain": ""}
    return {**c, "library": LIBV.get(c["class"], "—")}
