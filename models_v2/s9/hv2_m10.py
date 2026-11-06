"""Ж-214, лист 1: M10-величины дня SBER. Единицы %² (×1e4, как models_v2/daily.py). Все величины дня d — из баров основной сессии d
(begin <= 18:40, т.е. до 18:50 d); ничего из d+1. Каноническая RV, BV, J берутся из `daily.build_daily` (не пересчитываются).

Тождества (допуск hv2_paths.TOL_ID, относительный): rv_open + rv_mid + rv_close = rv; rs_pos + rs_neg = rv; c + j = rv, где
c := rv − j (= min(rv, bv)); BV не складывается с J отдельно (двойной счёт: BV + J = RV лишь при RV >= BV).
RQ = (n/3)·Σ r⁴ (оценка интегрированной квартичности), rv_se = sqrt(2·RQ/n) — асимптотическая ошибка измерения RV, rv_rel = rv_se / rv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import daily as D
from models_v2.s9 import hv2_sched as S

SC2 = D.SCALE ** 2


def bar_returns(bars: pd.DataFrame) -> pd.DataFrame:
    """Бары -> строки основной сессии: date, r (r_1 = ln(c/o), далее close-к-close внутри дня), begin, mins."""
    main = D.main_session(bars).sort_values("begin").reset_index(drop=True)
    r = D.intraday_returns(main)
    r["begin"] = main["begin"].to_numpy()
    r["mins"] = S.minutes(main["begin"])
    return r


def _parts(r: pd.DataFrame, regime: pd.Series, part_fn) -> np.ndarray:
    return part_fn(r["mins"].to_numpy(), r["date"].map(regime).to_numpy(), r.groupby("date").cumcount().to_numpy())


def aggregate(r: pd.DataFrame, regime: pd.Series, part_fn=S.part_by_clock) -> pd.DataFrame:
    """Строки баров -> дневные суммы (%²): части дня, знак, RQ."""
    x = r["r"].to_numpy(float)
    q = x * x
    part = _parts(r, regime, part_fn)
    f = pd.DataFrame({"date": r["date"], "rv": q, "rs_pos": np.where(x > 0, q, 0.0), "rs_neg": np.where(x < 0, q, 0.0),
                      "r4": q * q, "rv_open": np.where(part == S.OPEN, q, 0.0), "rv_mid": np.where(part == S.MIDDLE, q, 0.0),
                      "rv_close": np.where(part == S.CLOSE, q, 0.0), "n_open": (part == S.OPEN).astype(int),
                      "n_mid": (part == S.MIDDLE).astype(int), "n_close": (part == S.CLOSE).astype(int), "n_ret": 1})
    g = f.groupby("date").sum()
    sc = ["rv", "rs_pos", "rs_neg", "rv_open", "rv_mid", "rv_close"]
    g[sc] *= D.SCALE
    g["rq"] = g["n_ret"] / 3.0 * g["r4"] * SC2
    return g.drop(columns="r4")


def decomposition(d: pd.DataFrame) -> pd.DataFrame:
    """Из канонических rv, bv, j: c = rv − j (точно), доля скачка."""
    out = pd.DataFrame(index=d.index)
    out["bv"], out["j"] = d["bv"], d["j"]
    out["c"] = d["rv"] - d["j"]
    out["bv_gt_rv"] = d["bv"] > d["rv"]
    return out


def build(bars: pd.DataFrame, part_fn=S.part_by_clock, rq_lag: int = 0) -> pd.DataFrame:
    """bars SBER -> таблица дня на канонических датах (daily.build_daily). rq_lag — только для тестов-мутаций (RQ дня d+rq_lag); боевое значение 0."""
    d, _ = D.build_daily(bars)
    r = bar_returns(bars)
    sched = S.day_schedule(D.main_session(bars))
    a = aggregate(r, sched["regime"], part_fn).reindex(d.index)
    out = pd.DataFrame(index=d.index)
    out["n_bars"] = d["n_bars"]
    out["rv"] = d["rv"]
    out = out.join(decomposition(d).drop(columns=["bv", "j"]))
    out["bv"], out["j"] = d["bv"], d["j"]
    out = out.join(a.drop(columns=["rv", "n_ret"]))
    rq = a["rq"].shift(-rq_lag) if rq_lag else a["rq"]
    out["rq"] = rq
    out["rv_se"] = np.sqrt(2.0 * rq / d["n_bars"])
    out["rv_rel"] = out["rv_se"] / d["rv"]
    out["rv_sum_bars"] = a["rv"]
    out["regime_last_min"] = sched["regime"].reindex(d.index)
    out["sched_ok"] = sched["ok"].reindex(d.index)
    return out


def identities(t: pd.DataFrame) -> dict:
    """Максимальные относительные отклонения тождеств (к rv)."""
    rel = lambda a, b: float((np.abs(a - b) / b).max())  # noqa: E731
    return {"parts_sum": rel(t["rv_open"] + t["rv_mid"] + t["rv_close"], t["rv"]),
            "sign_sum": rel(t["rs_pos"] + t["rs_neg"], t["rv"]),
            "c_plus_j": rel(t["c"] + t["j"], t["rv"]),
            "bars_vs_canon": rel(t["rv_sum_bars"], t["rv"]),
            "bv_plus_j_wrong": rel(t["bv"] + t["j"], t["rv"])}
