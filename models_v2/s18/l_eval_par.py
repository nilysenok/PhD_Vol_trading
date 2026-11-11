"""S-L-EVAL (`S_L_SPEC.md`, разд. 2.5, 5.5): диагностика — â и λ* по месяцам и годам, разложение «замороженный π + реакция слоя», предсказания P-L1…P-L4 (выводятся кодом из таблиц; пороги — константы `l_eval_tab`, из спецификации). Только расчёт по сохранённым рядам.
Публичное API: `mech(frames, F, rv) -> DataFrame(section, series, item, value)`; `predictions(mon, yrs, F) -> DataFrame(id, statement, value, threshold, fulfilled)`."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s18 import l_eval_tab as LT

YEARS_PL4 = ("2021", "2025", "2026")


def _dpct(la, lb) -> float:
    return 100.0 * float(la.mean() - lb.mean()) / float(lb.mean())


def mech(frames: dict, F: dict, rv) -> pd.DataFrame:
    rows, lw = [], LT.ql(rv, F["W5-G"])
    for k, f in frames.items():
        tot, fr = _dpct(LT.ql(rv, f["F"].to_numpy(float)), lw), _dpct(LT.ql(rv, f["F_fr"].to_numpy(float)), lw)
        rows += [("decomp", k, "F_vs_W5G_pct", tot), ("decomp", k, "Ffr_vs_W5G_pct", fr), ("decomp", k, "reaction_pct", tot - fr)]
        m = f[f["active"]].groupby("refit_R", sort=True).first().reset_index()
        yr = pd.DatetimeIndex(m["refit_R"]).year
        rows += [("months", k, "n_active", len(m)), ("months", k, "n_a_zero", int((m["a"] == 0.0).sum())), ("months", k, "n_lam_pos", int((m["lam"] > 0).sum()))]
        for y in sorted(set(yr)):
            g = m[yr == y]
            pos = g[g["lam"] > 0]
            rows += [("a_by_year", k, "%d|n_months" % y, len(g)), ("a_by_year", k, "%d|n_lam_pos" % y, len(pos)), ("a_by_year", k, "%d|a_mean" % y, float(pos["a"].mean()) if len(pos) else float("nan")),
                     ("a_by_year", k, "%d|a_median" % y, float(pos["a"].median()) if len(pos) else float("nan"))]
        pos = m[m["lam"] > 0]
        rows += [("months", k, "a_median_lam_pos", float(pos["a"].median()) if len(pos) else float("nan")), ("months", k, "share_a_neg_lam_pos", float((pos["a"] < 0).mean()) if len(pos) else float("nan"))]
    return pd.DataFrame(rows, columns=["section", "series", "item", "value"])


def _v(m: pd.DataFrame, series: str, item: str) -> float:
    return float(m[(m["series"] == series) & (m["item"] == item)]["value"].iloc[0])


def predictions(mech_t: pd.DataFrame, yrs: pd.DataFrame, F: dict) -> pd.DataFrame:
    sh, med = _v(mech_t, "S-RL1", "share_a_neg_lam_pos"), _v(mech_t, "S-RL1", "a_median_lam_pos")
    y = yrs[(yrs["series"] == "S-RL1") & (yrs["base"] == "W5-G")].set_index("period")["contrib_pct"]
    c4 = [float(y[p]) for p in YEARS_PL4]
    same = bool(np.array_equal(F["S-RL1a"], F["W5-G"]))
    rows = [("P-L1", "â < 0 не менее чем в 90 %% месяцев с λ* > 0", sh, LT.PL1, bool(sh >= LT.PL1)),
            ("P-L2", "медиана â по месяцам с λ* > 0 в [%.3f; %.3f]" % LT.PL2, med, "%.3f…%.3f" % LT.PL2, bool(LT.PL2[0] <= med <= LT.PL2[1])),
            ("P-L3", "S-RL1a ≡ W5-G побитно", float(same), 1.0, same),
            ("P-L4", "вклад S-RL1 против W5-G в 2021, 2025, 2026 ≥ 0 п. п. (%s)" % ", ".join("%s: %+.4f" % (p, v) for p, v in zip(YEARS_PL4, c4)), min(c4), 0.0, bool(min(c4) >= 0))]
    return pd.DataFrame([dict(id=i, statement=s, value=v, threshold=t, fulfilled="выполнено" if ok else "нет") for i, s, v, t, ok in rows])
