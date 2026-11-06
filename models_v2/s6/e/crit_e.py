"""Описание строки волны E из ряда прогнозов: QLIKE по срезам, RMSE/MAE/R², crit_wf (R2), DM к ссылкам, проверка ACCEPT-пунктов (R7). Только из сохранённых рядов."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s5 import stats
from models_v2.s5.b1 import evalrows
from models_v2.s6.xgb import d_comp

YEARS = tuple(range(2017, 2027))
SLICES = (("all", "qlike_all"), ("p2017_2021", "qlike_2017_2021"), ("y2022", "qlike_2022"), ("p2023_2025", "qlike_2023_2025"),
          ("y2026", "qlike_2026"), ("no_240222", "qlike_no_240222"), ("no_2022", "qlike_no_2022"))


def qlike_series(rv, s2) -> np.ndarray:
    return np.asarray(metrics.qlike(rv, s2), float)


def cw(q, td) -> float:
    return d_comp.crit_wf(q, td)


def describe(name: str, s2, base: pd.DataFrame, refs: dict | None = None) -> dict:
    """base: d, target_date, rv. refs: {метка: σ̂² опоры} — DM (лаг 5, весь период и без 24.02.2022): d = QLIKE(строка) − QLIKE(опора), минус — строка лучше."""
    rv, td, s2 = base["rv"].to_numpy(float), base["target_date"], np.asarray(s2, float)
    if not (np.isfinite(s2).all() and (s2 > 0).all()):
        raise ValueError("%s: σ̂² не положительна/конечна" % name)
    q, mk = qlike_series(rv, s2), evalrows.masks(td)
    out = {"id": name, "n": len(rv), **{col: float(q[mk[k]].mean()) for k, col in SLICES}}
    out["qlike_2022_2025"] = float(q[mk["y2022"] | mk["p2023_2025"]].mean())
    out["crit_wf"] = cw(q, td)
    fm = stats.full_metrics(rv, s2)
    out.update({k: fm[k] for k in ("rmse_lvl", "mae_lvl", "r2_mz_lvl", "rmse_ln", "mae_ln", "bias_ln", "mean_r")})
    for lab, ref in (refs or {}).items():
        for suf, key in (("", "all"), ("_no_240222", "no_240222")):
            r = evalrows.vs(rv, np.asarray(ref, float), s2, mk[key])
            out.update({"d_%s%s" % (lab, suf): r["d"], "dm_%s%s" % (lab, suf): r["dm"], "p_%s%s" % (lab, suf): r["p"]})
    return out


def year_exclusion(q_new, q_base, td, years=YEARS) -> dict:
    """Пункт (3) R7: для каждого года k — среднее по датам вне года k разности QLIKE(новая) − QLIKE(опора); выигрыш = < 0."""
    y, d = pd.Series(td).dt.year.to_numpy(), np.asarray(q_new, float) - np.asarray(q_base, float)
    diffs = {int(k): float(d[y != k].mean()) for k in years}
    return {"diffs": diffs, "wins": int(sum(v < 0 for v in diffs.values()))}


def accept(new: dict, base: dict, q_new, q_base, td, seeds_new, seeds_base, crit_new: float, crit_base: float, min_wins: int = 9, tol_2017: float = 1.01) -> dict:
    """R7 (1)–(4): crit_new/crit_base — внутренний критерий (crit_pen или crit_wf); seeds_* — QLIKE по seed (None — не проверено => ACCEPT невозможен)."""
    c1 = bool(crit_base - crit_new > 1e-12)
    c2 = bool(new["qlike_2017_2021"] <= tol_2017 * base["qlike_2017_2021"])
    ye = year_exclusion(q_new, q_base, td)
    c3 = ye["wins"] >= min_wins
    c4 = None if (seeds_new is None or seeds_base is None) else bool(np.mean(seeds_new) < np.mean(seeds_base))
    ok = bool(c1 and c2 and c3 and c4 is True)
    return {"c1_crit": c1, "c2_2017_2021": c2, "c3_years": c3, "years_won": ye["wins"], "c4_seeds": c4, "accept": ok,
            "verdict": "ACCEPT" if ok else ("REJECT (пункт 4 не проверен)" if c4 is None and c1 and c2 and c3 else "REJECT")}
