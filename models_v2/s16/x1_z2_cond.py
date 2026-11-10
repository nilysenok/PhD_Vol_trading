"""Z2, условие запуска L1–L3 (план Z2 раздел 6; записано до результатов Z, не меняется) — механически, из таблиц Z (`Z_CRIT.csv`, `Z_PAIRS.csv`, `Z_STATES.csv`) и H0/HD. Чистые функции.
L1: ZB или ZC выполнила K1, K2, K5. L2: (а) ZB < ZB0 за весь период и в состоянии E; или (б) HD: контраст < 0, hi90 < 0 за весь период и тот же знак на 2018–2021.
L3 (все): в КАЖДОМ окне ступени 1 с 2018-07 ≥ 100 событий; LR H1-ступени 1 против пуассоновской базы на 2015–2021 p < 0,01; правило дубля не сработало (|corr| ξ_d ≤ 0,8 со всеми девятью). Условие ложно — H1 и Hc не считаются."""
from __future__ import annotations

import pandas as pd

MIN_EV, P_LIM, DUP_LIM = 100, 0.01, 0.8


def l1(crit: pd.DataFrame) -> bool:
    for y in ("ZB", "ZC"):
        c = crit[crit["y"] == y].set_index("crit")["passed"]
        if all(bool(c.get(k, False)) for k in ("K1", "K2", "K5")):
            return True
    return False


def l2a(pairs: pd.DataFrame, states: pd.DataFrame) -> bool:
    p = pairs[(pairs["y"] == "ZB") & (pairs["base"] == "ZB0") & (pairs["window"] == "all")].iloc[0]
    s = states[(states["state"] == "E") & (states["window"] == "all")].set_index("y")["y_qlike"]
    return bool(p["delta_q"] < 0 and s["ZB"] < s["ZB0"])


def l3(monthly: pd.DataFrame, lr_full: dict, dup: pd.DataFrame) -> dict:
    return dict(min_events=int(monthly["nev"].min()), ok_events=bool((monthly["nev"] >= MIN_EV).all()), p_strict=float(lr_full["p_strict"]), p_mix=float(lr_full["p_mix"]), ok_lr=bool(lr_full["p_strict"] < P_LIM and lr_full["p_mix"] < P_LIM),
                max_corr=float(dup["abs_corr"].max()), max_corr_var=str(dup.loc[dup["abs_corr"].idxmax(), "var"]), corr_N=float(dup.set_index("var").loc["N_d", "abs_corr"]), ok_dup=bool((dup["abs_corr"] <= DUP_LIM).all()))


def combine(a1: bool, a2: bool, a3: bool) -> dict:
    """-> {value, reasons}: условие истинно только при L1 ∧ L2 ∧ L3; reasons — ложные пункты."""
    r = [n for n, ok in (("L1", a1), ("L2", a2), ("L3", a3)) if not ok]
    return dict(value=not r, reasons=r)


def condition(crit, pairs, states, hd_neg: bool, monthly, lr_full, dup) -> dict:
    a, b, c = l1(crit), l2a(pairs, states), l3(monthly, lr_full, dup)
    a3 = c["ok_events"] and c["ok_lr"] and c["ok_dup"]
    detail = [n for n, ok in (("L3: окно с числом событий < %d (минимум %d)" % (MIN_EV, c["min_events"]), c["ok_events"]), ("L3: LR p ≥ %g" % P_LIM, c["ok_lr"]),
                              ("L3: дубль, |corr| = %.3f с %s > %g" % (c["max_corr"], c["max_corr_var"], DUP_LIM), c["ok_dup"])) if not ok]
    out = combine(a, b or hd_neg, a3)
    return dict(L1=a, L2a=b, L2b=bool(hd_neg), L2=bool(b or hd_neg), L3=bool(a3), L3_detail=c, value=out["value"], reasons=out["reasons"], l3_reasons=detail)


def table(c: dict) -> pd.DataFrame:
    d = c["L3_detail"]
    rows = [("L1", "ZB или ZC: K1, K2, K5", c["L1"], ""), ("L2а", "ZB < ZB0 за весь период и в состоянии E", c["L2a"], ""), ("L2б", "HD: контраст < 0, hi90 < 0, тот же знак на 2018–2021", c["L2b"], ""), ("L2", "(а) или (б)", c["L2"], ""),
            ("L3.1", "в каждом окне ≥ %d событий" % MIN_EV, d["ok_events"], "минимум %d" % d["min_events"]), ("L3.2", "LR p < %g (2015–2021)" % P_LIM, d["ok_lr"], "p_strict %.3g, p_mix %.3g" % (d["p_strict"], d["p_mix"])),
            ("L3.3", "дубль: |corr| ≤ %g" % DUP_LIM, d["ok_dup"], "максимум %.3f с %s; N_d %.3f" % (d["max_corr"], d["max_corr_var"], d["corr_N"])), ("L3", "все три", c["L3"], ""),
            ("условие", "L1 ∧ L2 ∧ L3", c["value"], "ложные: " + (", ".join(c["reasons"] + c["l3_reasons"]) or "нет"))]
    return pd.DataFrame(rows, columns=["item", "text", "value", "detail"])
