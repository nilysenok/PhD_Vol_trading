"""V4-DX-CS, G27: диагностика холодного старта (Ж-225 п. 3) — ЗАДНИМ ЧИСЛОМ, описательно, в прогноз и выбор не идёт (код `v4_dx_*` не импортируется путями прогноза, тест изоляции).
Потери от старта w = 0,5 в 2017 (цель t⁺ в 2017, общие даты S.common, M16 — среднее по бумагам, SBER отдельно): QLIKE C8 против (а) C8 с параметрами конца 2017 (оценка в первый торговый день 2018 по парам t⁺ < d0,
применена ко всем строкам 2017 — верхняя граница потерь холодного старта: параметры видят цели 2017), (б) лучшего постоянного веса на M за 2017 задним числом (геометрическая смесь, один вес на M16).
«Лучший допустимый приор по данным до 2017» — строка «нет данных»: OOF K2/K3 за 2016 нет, псевдоистория не создаётся."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from models_v2.s11 import v4_b_stats as ST, v4_c_run as CR, v4_cb_models as MD, v4_cb_run as CB, v4_io as IO

SERIES = ["k1", "k2", "k3", "d2", "C1", "C4", "C8", "CS1", "CS2"]
NO_PRIOR = "нет данных: OOF K2/K3 за 2016 не существует (данные начинаются в 2017, прогрев 2016 — только HAR-J); псевдоистория до 2017 не создаётся"


def end2017_forecast(P: pd.DataFrame, Zl: pd.DataFrame) -> pd.DataFrame:
    """(а): C8 с оценкой параметров в первый торговый день 2018 (закрытые пары t⁺ < d0, вся панель), применённой к строкам 2017. -> ticker, date, F."""
    P2, X = CR._prep(P, Zl, CB.SPECS["C8"][1])
    tickers = sorted(P2["ticker"].unique())
    tix = P2["ticker"].map({t: i for i, t in enumerate(tickers)}).to_numpy()
    d0 = P2.loc[P2["date"].dt.year == 2018, "date"].min()
    s, _, _ = CB._fit_period("regime", P2, X, tix, tickers, d0, None, None)
    sel = (P2["date"].dt.year == 2017).to_numpy()
    F = MD.predict("regime", s, {**CB._arr(P2, X, sel), "tix": tix[sel]})[0]
    return pd.DataFrame({"ticker": P2["ticker"].to_numpy()[sel], "date": P2["date"].to_numpy()[sel], "F_a": F})


def best_const_weight(rv, h, k2, k3) -> tuple:
    """(б): вес w ∈ [0, 1] на M = √(K2·K3) в F = H^(1−w)·M^w, минимизирующий средний QLIKE на строках (выпуклая задача по w). -> (w, F)."""
    rv, lH = np.asarray(rv, float), np.log(np.asarray(h, float))
    ell = 0.5 * (np.log(np.asarray(k2, float)) + np.log(np.asarray(k3, float))) - lH
    f = lambda w: float(np.mean(rv * np.exp(-(lH + w * ell)) + lH + w * ell))
    w = float(minimize_scalar(f, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-10}).x)
    return w, np.exp(lH + w * ell)


def _group_q(c: pd.DataFrame, col: str, group: str) -> float:
    s = c[c["ticker"] == "SBER"] if group == "SBER" else c[c["group"] == "M16"]
    return float(np.mean([ST.metr(g["rv_tplus"], g[col])["qlike"] for _, g in s.groupby("ticker")]))


def cold_start(r: dict) -> pd.DataFrame:
    """CS_DIAG.csv: по группе (M16, SBER) QLIKE 2017 каждого ряда, (а), (б), потери холодного старта (C8 − (а), C8 − (б)) в абсолютных значениях и % уровня K1; строка «приор до 2017 — нет данных»."""
    c = r["c"][r["c"]["t_plus"].dt.year == 2017].merge(end2017_forecast(r["P"], r["Zl"]), on=["ticker", "date"], how="left")
    m16 = c[c["group"] == "M16"]
    w_b = best_const_weight(m16["rv_tplus"], m16["k1"], m16["k2"], m16["k3"])[0]
    c = c.assign(F_b=np.exp(np.log(c["k1"].to_numpy(float)) + w_b * 0.5 * np.log(c["k2"].to_numpy(float) * c["k3"].to_numpy(float) / c["k1"].to_numpy(float) ** 2)))
    rows = []
    for g in ("M16", "SBER"):
        q = {s: _group_q(c, s, g) for s in SERIES}
        q["a_C8_params_end2017"], q["b_best_const_weight_2017"] = _group_q(c, "F_a", g), _group_q(c, "F_b", g)
        for k, v in q.items():
            rows.append({"group": g, "item": k, "qlike_2017": v, "d_to_c8": v - q["C8"], "d_to_c8_pct_k1": 100.0 * (v - q["C8"]) / q["k1"], "w_const": w_b if k.startswith("b_") else np.nan,
                         "n_rows": int(((c["group"] == "M16") if g == "M16" else (c["ticker"] == "SBER")).sum()), "note": "верхняя граница: параметры видят цели 2017" if k.startswith(("a_", "b_")) else ""})
        for k, ref in (("loss_cold_start_a", "a_C8_params_end2017"), ("loss_cold_start_b", "b_best_const_weight_2017")):
            rows.append({"group": g, "item": k, "qlike_2017": q["C8"] - q[ref], "d_to_c8": np.nan, "d_to_c8_pct_k1": 100.0 * (q["C8"] - q[ref]) / q["k1"], "w_const": np.nan, "n_rows": np.nan, "note": "C8 − " + ref + " (плюс — холодный старт стоит)"})
    rows.append({"group": "M16", "item": "prior_before_2017", "qlike_2017": np.nan, "d_to_c8": np.nan, "d_to_c8_pct_k1": np.nan, "w_const": np.nan, "n_rows": 0, "note": NO_PRIOR})
    df = pd.DataFrame(rows)
    IO.write_csv(df, "CS_DIAG.csv", "потери холодного старта 2017 (Ж-225 п. 3): ЗАДНИМ ЧИСЛОМ, описательно, не в выбор; цель t⁺ в 2017, общие даты; M16 — среднее по бумагам; SBER отдельно; (а), (б) — верхняя граница")
    return df
