"""Сетки c22a4b3 (порядок перебора сохранён: он решает ничьи) — RM, exec, B, C, D.

S1–S4: V4 (`RM_GRIDS_V4`, `TREND_RM_GRID_V4`, V4:301-352); S5/S6: V41 (S5R:496-521,
S6R:435-450, склейка V41:195-199 / 415-419). Сетки не менять (задание владельца).
"""
from __future__ import annotations

import numpy as np

from fastbt.signals.interface import expand_grid

CATEGORY = {"S1": "Contrarian", "S2": "Contrarian", "S3": "Trend", "S4": "Trend",
            "S5": "Range", "S6": "Range"}
STRATEGY_NAMES = {"S1": "S1_MeanRev", "S2": "S2_Bollinger", "S3": "S3_Donchian",
                  "S4": "S4_Supertrend", "S5": "S5_PivotPoints", "S6": "S6_VWAP"}
V41_SIDS = ("S5", "S6")          # A/B/C/D из V41
A_TEST_YEARS = list(range(2020, 2027))
BCD_TEST_YEARS = list(range(2022, 2027))

RM_CONTRARIAN = dict(sl_mult=[1.0, 1.5, 2.0], tp_mult=[2.0, 3.0, 5.0], max_hold=[15, 20, 30],
                     breakeven_trigger=[None, 1.0], cooldown_bars=[0, 5], partial_exit=[False])
S5_RM_PIVOT = dict(sl_target=["next_level", "next_level_half"],
                   tp_target=["P", "next_R", "half_to_next"], buffer=[0, 0.1, 0.2],
                   max_hold=[5, 8, 10], breakeven=[False, True], cooldown_bars=[0, 5])
S5_RM_ATR = dict(sl_mult=[0.3, 0.5, 0.75], tp_mult=[0.3, 0.5, 0.75, 1.0], max_hold=[5, 8, 10],
                 breakeven_trigger=[None, 1.0], cooldown_bars=[0, 5], time_decay=[False, True])
S6_RM_ATR = dict(sl_mult=[0.75, 1.0, 1.5], tp_mult=[0.75, 1.0, 1.5], max_hold=[5, 8, 10],
                 breakeven_trigger=[None, 1.0], cooldown_bars=[0, 5], time_decay=[False, True])
S6_RM_VWAP = dict(sl_dev=[1.5, 2.0, 2.5, 3.0], max_hold=[5, 8, 10], breakeven=[False, True],
                  cooldown_bars=[0, 5])
EXEC_GRID = dict(consec_entry=[1, 2], min_hold=[0, 5], cooldown_bars=[0, 10],
                 min_strength=[0.0, 0.3, 0.5])
B_GRIDS = {
    "Contrarian": dict(k_sl=[0.5, 0.75, 1.0, 1.5, 2.0], ratio=[0.5, 1.0, 1.5, 2.0, 3.0],
                       k_be=[None, 0.5, 1.0, 1.5, 2.0], gamma_hold=[None, 0.5, 1.0],
                       horizon=["h1", "h5", "h22"]),
    "Trend": dict(k_sl=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0], k_trail=[0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0],
                  k_be=[None, 0.5, 1.0, 1.5, 2.0], gamma_hold=[None, 0.5, 1.0],
                  horizon=["h1", "h5", "h22"]),
}
B_GRIDS["Range"] = dict(B_GRIDS["Contrarian"])
B_BASE_HOLD = {"Contrarian": 20, "Trend": 30, "Range": 10}

C_THRESHOLDS = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
C_THRESHOLDS_V41 = [round(x, 2) for x in np.arange(0.20, 0.81, 0.05)]
C_RANGE_BANDS = [(0.3, 0.7), (0.25, 0.75), (0.2, 0.8), (0.35, 0.65), (0.15, 0.85),
                 (0.4, 0.6), (0.1, 0.9)]
C_RANGE_BANDS_V41 = C_RANGE_BANDS + [(0.45, 0.55)]
C_LOOKBACKS = [126, 252, 504]
# ранг σ в масках C — всегда доля s[t−L..t−1] строго меньше s_t, без текущего дня
# (s9_stats.past_rank; аудит FA, Н1; ручки нет — Ъ ~23:20; печати v1 c22a4b3 — docs/otchety/arhiv_do_audita/)
C_HORIZONS = ["h1", "h5", "h22"]
C_DIRECTION = ["OFF", "rise", "fall"]
C_TERM_FILTER = [False, True]
C_HYSTERESIS = [0, 10]
C_LIMITS_V4 = [(5.0, 10.0), (3.0, 6.0)]        # (экспозиция %, сделок/год)
C_LIMITS_V41 = [(2.0, 3.0), (1.0, 1.0)]

D_TARGET_VOLS = [0.05, 0.10, 0.15, 0.20, 0.30]
D_MAX_LEVS = [1.0, 2.0, 3.0]
D_GAMMA = [0.5, 1.0, 1.5]
D_VOL_FLOORS = [None, 0.10]
D_VOL_CAPS = [None, 0.60]
D_SMOOTH = [None, 5, 20]
D_HORIZONS = ["h1", "h5", "h22"]
D_INV_LOOKBACKS = [63, 126, 252]
D_GATE_THRESHOLDS = [0.3, 0.5, 0.7, 1.0]
# Ручки дизайна (fastbt.design; по умолчанию — как в коде c22a4b3)
C_DIRECTION_TREND = list(C_DIRECTION)
C_HYSTERESIS_TREND = list(C_HYSTERESIS)
D_STANDARD = True
D_INVERSE = True
ENS_K = 10
VOTE_FRAC = 0.5
D_CONT = False                 # непрерывный D (Moreira–Muir) вместо бинарного затвора
D_CONT_GAMMA = [1.0, 2.0]
# Доводка D (ночная задача 5, 26.10): режим затвора abs (код) | rel | hyst; окно медианы σ_pred;
# пороги rel (открыт при σ/медиана < θ); пары (θ_in, θ_out) hyst; минимум доли закрытых дней
D_GATE_MODE = "abs"
D_REL_WINDOW = 252
D_REL_THR = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
D_HYST_IN = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
D_HYST_GAP = [0.1, 0.2, 0.3, 0.5]
D_MIN_CLOSED = 0.0
TF = False                     # фильтр оборота (fastbt.turnover)
TF_HOLD = [3, 5]
TF_REENTRY = [0, 1, 3, 5, 10]


def trend_rm_grid():
    grid = []
    for isl in [2.0, 2.5, 3.0]:
        for be in [None, 1.0]:
            for cd in [0, 5]:
                for tt in ["fixed_atr", "chandelier"]:
                    for tn in [10, 15, 20]:
                        for tam in [2.0, 3.0, 4.0, 5.0]:
                            grid.append(dict(initial_sl_mult=isl, trail_type=tt, trail_n=tn,
                                             trail_atr_mult=tam, breakeven_thresh=be,
                                             cooldown_bars=cd, parabolic_step=0.02,
                                             parabolic_max=0.15))
                for ps in [0.01, 0.02]:
                    for pm in [0.10, 0.15]:
                        grid.append(dict(initial_sl_mult=isl, trail_type="parabolic_step",
                                         trail_n=10, trail_atr_mult=2.5, breakeven_thresh=be,
                                         cooldown_bars=cd, parabolic_step=ps, parabolic_max=pm))
    for isl in [2.0, 3.0]:
        for mh in [30, 50, 100, 200]:
            for cd in [0, 5]:
                grid.append(dict(initial_sl_mult=isl, trail_type="fixed_atr", trail_n=mh,
                                 trail_atr_mult=99.0, breakeven_thresh=None, cooldown_bars=cd,
                                 parabolic_step=0.02, parabolic_max=0.15))
    return grid


def _tag(grid, t):
    out = []
    for rm in grid:
        d = dict(rm)
        d["_rm_type"] = t
        out.append(d)
    return out


def rm_grid(sid: str):
    """Полная RM-сетка стратегии в порядке старого кода."""
    if sid in ("S1", "S2"):
        return expand_grid(RM_CONTRARIAN)
    if sid in ("S3", "S4"):
        return trend_rm_grid()
    if sid == "S5":
        return _tag(expand_grid(S5_RM_PIVOT), "pivot") + _tag(expand_grid(S5_RM_ATR), "atr")
    if sid == "S6":
        return _tag(expand_grid(S6_RM_ATR), "atr") + _tag(expand_grid(S6_RM_VWAP), "vwap")
    raise KeyError(sid)


def coarse_idx(sid: str, n_rm: int):
    """Прореживание фазы 1: V4 (>200 → ::3, >20 → ::2), V41 (>20 → ::2)."""
    if sid in V41_SIDS:
        step = 2 if n_rm > 20 else 1
    else:
        step = 3 if n_rm > 200 else (2 if n_rm > 20 else 1)
    return list(range(0, n_rm, step))


def rm_key(rm: dict) -> str:
    return str(sorted(rm.items()))


def sp_key(sp: dict):
    return tuple(sorted(sp.items()))


def exec_grid():
    return expand_grid(EXEC_GRID)


def b_grid(cat: str):
    return expand_grid(B_GRIDS[cat])
