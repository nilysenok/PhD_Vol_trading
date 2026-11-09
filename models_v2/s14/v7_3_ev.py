"""V7-3, G43: оценка конфигурации 8 (чистые функции от массивов; файлы не читаются, ничего не пишется). Метрики, Δ, DM, Холм, «go», стресс, MCS — `v7_1b_ev` (импорт); пары и правило 6.2 — `v7_2_ev.pairs_table` (импорт).
Ряды на ОДИНАКОВЫХ датах: K1, K2, K3, M, K8, V4:NTP2, V6:A2, V6:A0, V6:A0g и конфигурации 1–8 (1–7 — версия 2). Сравнения 8: против 6 (вклад якоря; правило 6.2; стресс — без 1 % крайних дат пары и без 2018);
против K2, K3, M, K8, NTP2, A2, A0g (и K1, A0 — как в 2b) — Δ в % уровня K1 с 90 % интервалом и DM; «go» 6.1. Веса π и w — по годам цели (в среднем по строкам с переоценкой). Холм N = 733 / N_wide = 3166 — по окну."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s14 import v7_1b_ev as EV, v7_2_ev as E2, v7_2_evspec as SP

CFGS = E2.CFGS + ("cfg8",)
PAIRS8 = (("H8 якорь", "cfg8", "cfg6"), ("8 против 4 (справочно)", "cfg8", "cfg4"))        # первая — основная (правило 6.2); вторая — справочно (Δ к 4 с интервалом и DM, без вывода)
QUANT = ("pi0", "pi1", "pi2", "w")
PERIODS = SP.PERIODS


def weights_table(df: pd.DataFrame) -> pd.DataFrame:
    """Среднее π_0, π_1, π_2 и w по годам цели 2018…2026 и до / после 2021; только строки с переоценкой (fitted > 0)."""
    d = df[df["fitted"] > 0]
    y = pd.DatetimeIndex(d["t_plus"]).year.to_numpy()
    sel = [(str(yy), y == yy) for yy in range(2018, 2027)] + [(PERIODS[-2], y <= 2021), (PERIODS[-1], y >= 2022)]
    return pd.DataFrame([dict(config="cfg8", quantity=q, period=n, n=int(m.sum()), mean=float(d[q].to_numpy(float)[m].mean()) if m.any() else float("nan")) for q in QUANT for n, m in sel])


def main_tables(rv, s: dict, tp, df8: pd.DataFrame) -> dict:
    """METRICS, MCS, PAIRS (8 против 6), TESTS / GO / YEARS (конфигурация 8), STRESS, WEIGHTS."""
    refs = {k: s[k] for k in EV.REFS}
    t = {"METRICS": EV.metrics_table(rv, s, tp), "MCS": EV.mcs_table(rv, s, tp), "PAIRS": E2.pairs_table(rv, s, tp, PAIRS8),
         "TESTS": EV.tests_table(rv, s["cfg8"], refs, tp).assign(config="cfg8"), "GO": EV.go_table(rv, s["cfg8"], refs, tp).assign(config="cfg8"),
         "YEARS": EV.years_table(rv, s["cfg8"], refs, tp).assign(config="cfg8"), "STRESS": EV.stress_table(rv, {k: s[k] for k in E2.CTRL + CFGS}, tp)}
    t["WEIGHTS"] = weights_table(df8)
    return t
