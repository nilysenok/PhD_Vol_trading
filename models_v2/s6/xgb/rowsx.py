"""Строка таблиц волны B потока XGB: метрики из ряда прогнозов (P6) + внутренний критерий (P1) + перспективность (P2) + устойчивость (P5)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s5.b1 import evalrows
from models_v2.s6.xgb import cfgx, critx, stabx

STATUS = ("DEVELOPMENT (выбор по внутреннему LOYO-критерию); HISTORICAL WALK-FORWARD (ряд прогнозов 2017–02.2026, уже просмотрено, не независимый OOS); "
          "p — сырые, множественный поиск не учтён (потолок)")
NOTE = "волна B Ж-206, поток XGB; " + STATUS


def write_csv(df: pd.DataFrame, path, note: str) -> None:
    """Таблица с шапкой-комментарием: статус и примечание (статус в каждой таблице)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# %s; %s\n" % (NOTE, note))
        df.to_csv(f, index=False)


def with_f4(fc: pd.DataFrame, ctrl: pd.DataFrame, f4_s2: np.ndarray) -> dict:
    """Разность и DM к опорной строке Б-2+F4 (d = QLIKE строки − QLIKE опоры: минус — строка лучше), весь период и без 24.02.2022."""
    rv, s2 = fc["rv"].to_numpy(float), fc["sigma2"].to_numpy(float)
    mk = evalrows.masks(fc["target_date"])
    out = {}
    for lab, key in (("", "all"), ("_no_240222", "no_240222")):
        r = evalrows.vs(rv, np.asarray(f4_s2, float), s2, mk[key])
        out.update({"d_f4%s" % lab: r["d"], "dm_f4%s" % lab: r["dm"], "p_f4%s" % lab: r["p"]})
    return out


def row_of(c: dict, tid: str, fc: pd.DataFrame, folds: pd.DataFrame, ctx: dict, prior: tuple | None = None) -> dict:
    """c — конфигурация; ctx: ctrl (s2_b2, s2_harj), f4_s2 (σ̂² опоры), crit_b2, crit_f4 (dict crit)."""
    cr = critx.crit(folds)
    r = {"id": c["name"], "trial_id": tid, "transferred": False, "source": "волна B Ж-206 (новое)", "direction": c["direction"], "objective": c["loss"],
         "weights_gamma": c["gw"], "features": "Б-2+" + "+".join(c["groups"]) + (" без " + c["drop"] if c["drop"] else "")}
    r.update(evalrows.row(fc[["d", "target_date", "rv", "sigma2"]], ctx["ctrl"]))
    r.update(with_f4(fc, ctx["ctrl"], ctx["f4_s2"]))
    r.update(crit_mean=cr["crit_mean"], crit_pen=cr["crit_pen"], crit_pen_b2=ctx["crit_b2"]["crit_pen"], crit_pen_f4=ctx["crit_f4"]["crit_pen"])
    r["unstable"] = stabx.unstable(fc, ctx["f4_s2"])
    r["perspective"] = critx.perspective(cr["crit_pen"], ctx["crit_b2"]["crit_pen"], r["d_harj_no_240222"], r["unstable"])
    r["beats_f4_ref"] = bool(critx.better(cr["crit_pen"], ctx["crit_f4"]["crit_pen"]) and not r["unstable"])
    r["n_dates"], r["status"] = len(fc), STATUS
    return r


def trial_record(c: dict, tid: str, r: dict, secs: float, stamp: str) -> dict:
    """Строка реестра OPTUNA_TRIALS.csv (шапка cfgx.TRIAL_HEAD)."""
    keep = ("qlike_all", "qlike_2017_2021", "qlike_2022", "qlike_2023_2025", "d_harj_no_240222", "d_b2", "d_f4")
    return {"trial_id": tid, "wave": "B-xgb", "direction": c["direction"], "config": str(cfgx.canon(c)),
            "metric_inner": "crit_pen=%.6f;crit_mean=%.6f;%s" % (r["crit_pen"], r["crit_mean"], ";".join("%s=%.6f" % (k, r[k]) for k in keep)),
            "seed": 42, "time": "%s;%.1fs" % (stamp, secs)}
