"""S-RCD-EVAL (Ж-259; `S_RCD_SPEC.md`, разд. 6.3, 8.4): параметры и выбор по месяцам S-RCD, S-CDF, S-CKF (и S-RCK для сравнения). Читает СОХРАНЁННЫЕ `RD/*.parquet`, `R/rck.parquet`; ничего не обучает и не пишет.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Исследовательский результат на раскрытой истории, не OOS.
Публичное API:
- `FORM` — ряд → форма (CD / CK); `params(frames) -> long(variant, kind, item, key, value)`: месяцы (активных, θ = 0, доля θ ≠ 0), число месяцев по λ*, c, m, κ (R), ‖θ‖₀ по месяцам (min / медиана / max), доля месяцев с параметром на границе формы,
  разброс L по перезапускам (Н4: медиана и максимум), квантили θ (месяцы с θ ≠ 0);
- `sel(frames) -> DataFrame`: по месяцам: variant, refit_R, n_train, c, m, κ, λ*, w_λ, G_λ,b (R-ряды), val_* (S-CDF, S-CKF), spread."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s18 import cm_model as CM
from models_v2.s18 import r_eval_par as RP

FORM = {"S-RCD": "CD", "S-CDF": "CD", "S-CKF": "CK", "S-RCK": "CK"}
QS = (0.05, 0.5, 0.95)
EPS = 1e-9
_row, _months = RP._row, RP._months
SELCOLS = ["variant", "refit_R", "n_train", "c", "m", "kappa", "lam", "w_1e-1", "w_1e-2", "w_1e-3"] + ["G_%s_b%d" % (k, b) for k in ("1e-1", "1e-2", "1e-3") for b in (1, 2, 3)] + ["val_1e-4", "val_1e-3", "val_1e-2", "spread"]


def _theta(v: str, df: pd.DataFrame, m: pd.DataFrame, nz: pd.DataFrame) -> list:
    names, out = CM.NAMES[FORM[v]], []
    if len(nz):
        l0 = (np.abs(nz[list(names)].to_numpy(float)) > 0).sum(axis=1)
        out += [_row(v, "l0", n, "", float(f(l0))) for n, f in (("min", np.min), ("median", np.median), ("max", np.max))]
        for p, (lo, hi) in zip(names, CM.bounds(FORM[v])):
            at = (np.abs(nz[p] - lo) < EPS) | (np.abs(nz[p] - hi) < EPS)
            out += [_row(v, "at_bound", p, "[%g; %g]" % (lo, hi), float(at.mean()))] + [_row(v, "theta_q", p, "q%02d" % round(100 * q), float(np.quantile(nz[p], q))) for q in QS]
    return out


def params(frames: dict) -> pd.DataFrame:
    out = []
    for v in FORM:
        df = frames[v]
        m = _months(df)
        r_proto = v in ("S-RCD", "S-RCK")
        nz = m[m["lam"] > 0]
        out += [_row(v, "months", "active", "", len(m)), _row(v, "months", "theta_zero", "", int((m["lam"] == 0.0).sum())), _row(v, "months", "share_theta_nonzero", "", len(nz) / len(m))]
        out += [_row(v, "lam", "%g" % lam, "", int((m["lam"] == lam).sum())) for lam in sorted(m["lam"].unique())]
        if r_proto:
            kap = m["c"] / m["m"]
            out += [_row(v, "clip", n, "", float(f(x))) for n, x, f in (("c_median", m["c"], np.median), ("c_min", m["c"], np.min), ("c_max", m["c"], np.max), ("m_median", m["m"], np.median), ("kappa_median", kap, np.median))]
        else:
            out += [_row(v, "n_small_pool", "lam_1e-2_below_252", "", int((m["n_train"] < 252).sum()))]
        if v != "S-RCK":
            sp = nz["spread"].dropna()
            out += [_row(v, "spread", n, "", float(f(sp))) for n, f in (("median", np.median), ("max", np.max))] if len(sp) else []
            out += _theta(v, df, m, nz)
    return pd.DataFrame(out)


def sel(frames: dict) -> pd.DataFrame:
    out = []
    for v in ("S-RCD", "S-CDF", "S-CKF"):
        m = _months(frames[v])
        t = m.copy()
        t["kappa"], t["variant"] = t["c"] / t["m"], v
        out.append(t.reindex(columns=SELCOLS))
    return pd.concat(out, ignore_index=True)
