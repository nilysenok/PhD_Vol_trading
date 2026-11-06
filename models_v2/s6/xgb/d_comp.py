"""Составные строки волны D (годовой выбор окна по прошлому, переключение R4) и причинный критерий частоты crit_wf (GATES_D_xgb Q3, Q4, Q6). Новых обучений нет."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s4 import rows
from models_v2.s5.b1 import designs

PENALTY = 0.25
RAW_COLS = ["f", "s2", "k", "T", "secs"]


def crit_wf(q, target_dates, penalty: float = PENALTY) -> float:
    """Причинный критерий ряда потерь: ann_y — средний QLIKE по году цели; pen_Y = mean_{y<Y} ann_y + 0,25·(max − mean), Y = 2018…; crit_wf = mean_Y pen_Y."""
    q, y = np.asarray(q, float), pd.Series(target_dates).dt.year.to_numpy()
    ann = {int(k): float(q[y == k].mean()) for k in sorted(set(y))}
    pens = []
    for Y in sorted(ann):
        past = [v for k, v in ann.items() if k < Y]
        if past:
            m = float(np.mean(past))
            pens.append(m + penalty * (max(past) - m))
    return float(np.mean(pens))


def freq_choice(cwf: dict, thr: float = 0.0005) -> str:
    """Q4: от самого редкого. Еженедельное заменяет ежемесячное, только если crit_wf(w) <= crit_wf(m) − thr; ежедневное — только если crit_wf(d) <= crit_wf(выбранного) − thr."""
    cur = "w" if cwf["w"] <= cwf["m"] - thr else "m"
    return "d" if cwf["d"] <= cwf[cur] - thr else cur


def stress_state(F) -> np.ndarray:
    """Состояние на строке t: stress_RV (RV_w > q67 по 252 предыдущим, без текущего); NaN (нет истории) — спокойно. Известно на t."""
    st = designs.regime_cols(F)["stress"]
    return np.where(np.isnan(st), 0.0, st).astype(bool)


def compose_raw(ra: pd.DataFrame, rb: pd.DataFrame, use_a: np.ndarray) -> pd.DataFrame:
    """Прогноз строки row — из A там, где use_a[row], иначе из B (use_a — массив по строкам F; состояние на row)."""
    if not (ra["row"].to_numpy() == rb["row"].to_numpy()).all():
        raise ValueError("сетки A и B не совпадают")
    pick = np.asarray(use_a)[ra["row"].to_numpy()]
    out = ra.copy()
    out.loc[~pick, RAW_COLS] = rb.loc[~pick, RAW_COLS].to_numpy()
    return out


def compose_folds(fa: pd.DataFrame, fb: pd.DataFrame, use_a: np.ndarray) -> pd.DataFrame:
    """Фолды (Y, y, q) составной строки: потеря проверяемой пары t — из A при use_a[t] (состояние на t известно в t), иначе из B."""
    m = fa.merge(fb, on=["Y", "y", "row"], suffixes=("_a", "_b"))
    if len(m) != len(fa) or len(m) != len(fb):
        raise ValueError("фолды A и B не совпадают")
    m["q"] = np.where(np.asarray(use_a)[m["row"].to_numpy()], m["q_a"], m["q_b"])
    return m.groupby(["Y", "y"], as_index=False)["q"].mean().sort_values(["Y", "y"]).reset_index(drop=True)


def annual_raw(F, raws: dict, choice: dict, default: str = "expanding") -> pd.DataFrame:
    """Прогноз строки row — из окна, выбранного для года цели (choice {Y: имя | None}; None — default). Выбор года Y сделан по целям < 01.01.Y."""
    base = raws[default]
    pieces = []
    for Y, name in sorted(choice.items()):
        src = raws[name or default]
        yr = np.array([rows.year_of(F, int(r)) for r in base["row"]])
        pieces.append(src[yr == Y])
    return pd.concat(pieces).sort_values("row").reset_index(drop=True)
