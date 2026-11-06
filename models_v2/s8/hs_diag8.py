"""Ж-212, этап A, вопросы 7–8. Q7: информация в разности прогнозов δ = ln HAR − ln ML (MZ-регрессия ошибки ML на δ, HAC). Q8 (главный):
условная ценность HAR сверх бустинга — QLIKE-согласованная модель σ̂² = ML·exp(X·γ), X ∈ {δ; ln RV_d/w/m − ln ML; ln(1+J); асимметрия RS−/RS+; все пять}.
Два измерения: (а) в выборке — Wald с HAC (потолок: видит всю историю); (б) OOS по прошлому — γ подогнана только по закрытым прогнозам (target_date <= d),
расширяющееся окно, 252 закрытых до старта, DM Ньюи–Уэста к самому бустингу. Это диагностика (не trial): правила выбора нет, спецификации объявлены заранее."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s3 import dm as dmod
from models_v2.s8 import hs_diag as H
from models_v2.s8 import hs_enc

BASES = {"XJ": "delta_HX", "LF": "delta_HL"}
STEP = 5


def q7_table(o: pd.DataFrame) -> pd.DataFrame:
    mk, rows = H.period_masks(o), []
    for ml, dcol in BASES.items():
        e, dl = o["e_" + ml].to_numpy(), o[dcol].to_numpy()
        for p, m in mk.items():
            r = hs_enc.mz_hac(e[m], dl[m])
            rh = hs_enc.mz_hac(o["e_HAR"].to_numpy()[m], -dl[m])
            ra = hs_enc.mz_hac(np.abs(e[m]), np.abs(dl[m]))
            rows.append({"ml": ml, "period": p, "n": int(m.sum()), "b_eML_on_delta": r["b"], "t": r["t_b"], "p": r["p_b"], "r2": r["r2"], "b_eHAR_on_minus_delta": rh["b"],
                         "t_eHAR": rh["t_b"], "b_abs_e_on_abs_delta": ra["b"], "t_abs": ra["t_b"]})
    dq, pm = o["dq_HAR_XJ"].to_numpy(), o["past_dqX_m22"].to_numpy()
    ok = np.isfinite(pm)
    r = hs_enc.mz_hac(dq[ok], pm[ok])
    rows.append({"ml": "persistence", "period": "all", "n": int(ok.sum()), "b_eML_on_delta": r["b"], "t": r["t_b"], "p": r["p_b"], "r2": r["r2"]})
    return pd.DataFrame(rows)


def spec_columns(o: pd.DataFrame, ml: str) -> dict:
    lb, dl = np.log(o[ml].to_numpy(float)), o[BASES[ml]].to_numpy(float)
    one = np.ones(len(o))
    comp = {"d": o["ln_rv_d"].to_numpy() - lb, "w": o["ln_rv_w"].to_numpy() - lb, "m": o["ln_rv_m"].to_numpy() - lb,
            "J": o["ln1p_J"].to_numpy(float), "asym": o["rs_asym"].to_numpy(float)}
    s = {"const": [one], "const+delta": [one, dl], "delta_only": [dl]}
    s.update({"const+" + k: [one, v] for k, v in comp.items()})
    s["const+all5"] = [one] + list(comp.values())
    s["const+d+w+m"] = [one, comp["d"], comp["w"], comp["m"]]
    return {k: np.column_stack(v) for k, v in s.items()}


def _oos_row(o, name, ml, X, base_s2, rv, d, td, mk) -> tuple:
    s2, G = hs_enc.oos_expanding(rv, base_s2, X, d, td, step=STEP)
    w = hs_enc.wald(rv, base_s2, X)
    row = {"spec": name, "base": ml, "k": X.shape[1], "gamma": ";".join("%.4f" % g for g in w["gamma"]), "t_hac": ";".join("%.2f" % t for t in w["t"]),
           "p_joint_in_sample": w["p_joint"], "gain_in_sample_ceiling": w["gain_in_sample"], "qlike_base": hs_enc.qlike_mean(rv, base_s2), "qlike_oos": hs_enc.qlike_mean(rv, s2)}
    for p, m in mk.items():
        r = hs_enc.dm_vs(rv, base_s2, s2, m)
        row.update({"delta_" + p: r["delta"], "p_" + p: r["p"]})
    return row, s2


def _by_year(o, name, ml, base, s2) -> list:
    rv, y, q = o["rv"].to_numpy(float), H.year(o), hs_enc.qlike_mean
    return [{"spec": name, "base": ml, "year": int(k), "qlike_base": q(rv[y == k], base[y == k]), "qlike_oos": q(rv[y == k], s2[y == k]),
             "delta": q(rv[y == k], s2[y == k]) - q(rv[y == k], base[y == k])} for k in sorted(set(y))]


def q8_tables(o: pd.DataFrame) -> tuple:
    """-> (таблица спецификаций, таблица OOS по годам). Обратное направление (бустинг сверх HAR): база HAR, X = [1, −δ]."""
    rv, d, td, mk = o["rv"].to_numpy(float), o["d"].to_numpy("datetime64[ns]"), o["target_date"].to_numpy("datetime64[ns]"), H.period_masks(o)
    jobs = [(name, ml, X, o[ml].to_numpy(float)) for ml in BASES for name, X in spec_columns(o, ml).items()]
    jobs += [("reverse: const+(-delta)", "HAR_beyond_" + ml, np.column_stack([np.ones(len(o)), -o[dc].to_numpy(float)]), o["HAR"].to_numpy(float)) for ml, dc in BASES.items()]
    rows, yrows = [], []
    for name, ml, X, base in jobs:
        row, s2 = _oos_row(o, name, ml, X, base, rv, d, td, mk)
        rows.append(row)
        yrows += _by_year(o, name, ml, base, s2)
    t = pd.DataFrame(rows)
    for c in ("all", "no_240222"):
        t["p_holm_" + c] = _holm(t["p_" + c].fillna(1.0))
    return t, pd.DataFrame(yrows)


def beta_profile(o: pd.DataFrame, grid=(-0.2, 0.0, 0.1, 0.2, 0.357, 0.5, 1.0)) -> pd.DataFrame:
    """Профиль по β для σ̂² = ML·exp(β·δ): QLIKE и sd логарифмической ошибки. β = 1 — HAR; β ≈ 0,36 — МНК-коэффициент MZ в логах (Q7).
    Показывает, что информация в δ есть для ln-ошибки, но не для QLIKE (разные функции потерь)."""
    rv, rows = o["rv"].to_numpy(float), []
    for ml, dcol in BASES.items():
        for b in grid:
            s2 = o[ml].to_numpy(float) * np.exp(b * o[dcol].to_numpy(float))
            rows.append({"ml": ml, "beta": b, "qlike": hs_enc.qlike_mean(rv, s2), "sd_ln_error": float(np.log(rv / s2).std()), "mean_ln_error": float(np.log(rv / s2).mean())})
    return pd.DataFrame(rows)


def _holm(p) -> np.ndarray:
    return dmod.holm(np.asarray(p, float))
