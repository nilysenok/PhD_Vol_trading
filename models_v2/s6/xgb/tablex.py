"""Таблицы потока XGB волны B: OBJECTIVE_RESULTS.csv и FEATURE_SETS.csv = перенос Ж-204 (без пересчёта) + новые строки волны B."""
from __future__ import annotations

import pandas as pd

from models_v2.s6.xgb import cfgx, rowsx, transferx

OBJ_PREFIX = ("x3_", "x4_", "x5_")
KEEP_NEW = ["crit_pen_b2", "crit_pen_f4", "weights_gamma", "beats_f4_ref", "perspective", "unstable", "d_harj", "dm_harj", "p_harj", "d_harj_no_240222",
            "dm_harj_no_240222", "p_harj_no_240222", "d_b2", "dm_b2", "p_b2", "d_b2_no_240222", "dm_b2_no_240222", "p_b2_no_240222", "d_f4", "dm_f4", "p_f4",
            "d_f4_no_240222", "dm_f4_no_240222", "p_f4_no_240222", "share_years_win_harj", "n_dates"]


def split_new(rows: dict) -> tuple:
    """(строки obj, строки feat) новых прогонов; ctrl_f4 — опорная строка в обеих таблицах."""
    obj, feat = [], []
    for k, r in rows.items():
        if k == "ctrl_f4" or k.startswith(OBJ_PREFIX):
            obj.append(r)
        if k == "ctrl_f4" or r["direction"] == "feature":
            feat.append(r)
    return obj, feat


def frame(transferred: pd.DataFrame, new: list) -> pd.DataFrame:
    cols = list(transferred.columns) + [c for c in KEEP_NEW if c not in transferred.columns]
    df = pd.DataFrame(new)
    df = df.reindex(columns=cols) if len(df) else pd.DataFrame(columns=cols)
    return pd.concat([transferred.reindex(columns=cols), df], ignore_index=True)


def no_crossing(df: pd.DataFrame) -> bool:
    """Ни одна строка не получает результат другой: id и trial_id внутри таблицы уникальны."""
    return bool(df["id"].is_unique and df["trial_id"].dropna().is_unique)


def stage(rows: dict) -> int:
    obj, feat = split_new(rows)
    fo, ff = frame(transferx.objectives(), obj), frame(transferx.features(), feat)
    bad = [n for n, d in (("OBJECTIVE_RESULTS", fo), ("FEATURE_SETS", ff)) if not no_crossing(d)]
    rowsx.write_csv(fo, cfgx.XGB_REP / "OBJECTIVE_RESULTS.csv", "X1–X5; transferred=true — Ж-204 на Б-2, без пересчёта; новые — на Б-2+F4; QLIKE в %²")
    rowsx.write_csv(ff, cfgx.XGB_REP / "FEATURE_SETS.csv", "F1–F6, комбинации, абляции; transferred=true — Ж-204, без пересчёта")
    print("B8: OBJECTIVE_RESULTS %d строк, FEATURE_SETS %d строк; дубли id/trial_id: %s" % (len(fo), len(ff), bad or "нет"))
    return 1 if bad else 0
