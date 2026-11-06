"""Ж-212 (HS11, G13): формальные уровни 1–4 для модели M. Опора H = HAR-J, S* = XJ (лучший самостоятельный бустинг по crit_wf), LF — тест C. Уровень 4 — «не проверялся» (решение о выборке не принято).
Δ = QLIKE(M) − QLIKE(опора), минус — M лучше. Метка «потолок» генерируется кодом: финалист выбран по crit_wf на всей истории, оценочные годы просмотрены."""
from __future__ import annotations

import numpy as np

from models_v2.s7 import hf_levels as V
from models_v2.s8 import hs_eval

PER3 = ("G_2019_2021", "P2_2022", "P3_2023_2025")
CRIT = 0.0005
REFS = ("HAR-J", "XGBoost", "LightGBM")


def ceiling_note(model: str) -> str:
    """Все финалисты, кроме трёх опор, выбраны по crit_wf на всей истории — потолок; XGB-строки B наследуют HP XJ, выбранные на 2017–21."""
    if model in REFS:
        return ""
    s = "потолок: финалист выбран по crit_wf на всей истории (2017–02.2026 просмотрена)"
    if model.startswith("B-XGB"):
        s += "; HP XGB = hp_by_year XJ, выбранные на 2017–21"
    if model.startswith(("B-", "E")):
        s += "; признаки/поправка используют ряды XJ/LF или C (потолок наследуется)"
    return s


def top5_wo(dl: np.ndarray) -> float:
    """Δ без 5 дней с наибольшим вкладом в преимущество (самые отрицательные дневные разности)."""
    keep = np.ones(len(dl), bool)
    keep[np.argsort(dl)[:5]] = False
    return float(dl[keep].mean())


def pick(t, model, test, period):
    return V.pick(t, model, test, period)


def levels(t, model: str, crit: dict, qall: dict, dl_star: np.ndarray, star: str = "XJ") -> dict:
    """Пункты уровней 1–3 по HS11; t — STATISTICAL_TESTS с p_holm; crit/qall — ключи 'HAR', 'S*' и имя модели; dl_star — QLIKE(M) − QLIKE(S*) по датам."""
    A = lambda p: pick(t, model, "A", p)
    B = lambda p: pick(t, model, "B", p)
    o = {}
    o["L1_a"] = bool(qall[model] < qall["HAR"] and crit[model] < crit["HAR"])
    o["L1_b"] = bool(sum(A(p)["delta"] < 0 for p in PER3) >= 2 and A("no_240222")["delta"] < 0)
    o["L1_c"] = bool(A("all")["p_raw"] < 0.05 and A("all")["boot_hi95"] < 0)
    o["L1_d"] = bool(A("all")["p_holm"] < 0.10)
    o["L1"] = bool(o["L1_a"] and o["L1_b"] and o["L1_c"] and o["L1_d"])
    o["L2_b"] = bool(crit[model] < crit["S*"] - CRIT and B("all")["delta"] < 0)
    o["L2_c"] = bool(B("all")["p_raw"] < 0.05 and B("all")["boot_hi95"] < 0 and B("all")["p_holm"] < 0.10)
    o["L2_d"] = bool(pick(t, model, "C", "all")["delta"] < 0)
    o["L2"] = bool(o["L1"] and o["L2_b"] and o["L2_c"] and o["L2_d"])
    o["L3_b"] = bool(sum(B(p)["delta"] < 0 for p in PER3) >= 2)
    o["L3_c"] = bool(B("no_240222")["delta"] < 0 and B("no_240222")["p_raw"] < 0.10)
    o["L3_d"] = bool(top5_wo(dl_star) < 0)
    o["L3_e"] = bool(B("late_2019_2026")["delta"] < 0)
    o["L3"] = bool(o["L2"] and o["L3_b"] and o["L3_c"] and o["L3_d"] and o["L3_e"])
    o["L4"] = "не проверялся"
    o["verdict"] = "L3" if o["L3"] else "L2" if o["L2"] else "L1" if o["L1"] else "L1 только сырой" if (o["L1_a"] and o["L1_b"] and o["L1_c"]) else "нет"
    o["ceiling_note"] = ceiling_note(model)
    return o


def mask_late(df) -> np.ndarray:
    return hs_eval.period_mask(df, 2019, 2026)
