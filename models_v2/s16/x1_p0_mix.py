"""X1-пилот, лист P0, G4: соответствует ли оператор смешивания математическому смыслу экспертов. Только чтение сохранённых прогнозов (X1, X2, V7 cfg6, V4 NTP2/C8, K1/K2/K3/K8); ничего не обучается и не пишется.
Оператор задан экспертами F_k > 0 (n × K) и весами π_k (n × K, строки в симплексе). Геометрический: F_geo = exp(Σπ_k ln F_k); арифметический: F_ar = Σπ_k F_k.
Йенсен для дневной QLIKE, потеря при RV = Y:
 · лог-пространство, f = ln F: l(f) = f + Y·e^{−f}, l'' = Y·e^{−f} > 0 — выпукла ⇒ l(Σπ f_k) ≤ Σπ l(f_k) ВСЕГДА; доля нарушений обязана быть 0 (иначе ошибка кода или данных);
 · по F: l_F(F) = ln F + Y/F, l_F'' = (2Y − F)/F³ — выпукла только при F < 2Y ⇒ l_F(Σπ F_k) ≤ Σπ l_F(F_k) может нарушаться (доля нарушений — число).
Смещение (iii): по неравенству AM–GM F_geo ≤ F_ar; mean(F_geo)/mean(F_ar) ≤ 1 — по годам. Единицы — %², как у RV."""
from __future__ import annotations

import numpy as np
import pandas as pd

TOL = 1e-12
YEARS = tuple(range(2018, 2026)) + (2026, "all")


def lq_log(f, y):
    return f + y * np.exp(-f)


def lq_lin(F, y):
    return np.log(F) + y / F


def check_op(rv, Fk, pi) -> dict:
    """-> по датам: gap_log = Σπ l(f_k) − l(Σπ f_k) (≥ 0 по Йенсену), gap_ar = Σπ l_F(F_k) − l_F(Σπ F_k) (знак любой), F_geo, F_ar."""
    rv, Fk, pi = np.asarray(rv, float), np.asarray(Fk, float), np.asarray(pi, float)
    f = np.log(Fk)
    fg = (pi * f).sum(1)
    Fa = (pi * Fk).sum(1)
    gl = (pi * lq_log(f, rv[:, None])).sum(1) - lq_log(fg, rv)
    ga = (pi * lq_lin(Fk, rv[:, None])).sum(1) - lq_lin(Fa, rv)
    return dict(gap_log=gl, gap_ar=ga, F_geo=np.exp(fg), F_ar=Fa)


def qlike(rv, F):
    r = np.asarray(rv, float) / np.asarray(F, float)
    return r - np.log(r) - 1.0


def rows_for(name: str, meaning: str, dates: pd.DatetimeIndex, rv, Fk, pi, F_saved=None) -> list:
    kind = "arith" if meaning.startswith("арифм") else "geo"
    c = check_op(rv, Fk, pi)
    out = []
    for y in YEARS:
        m = np.ones(len(dates), bool) if y == "all" else (dates.year == y)
        if not m.any():
            continue
        s = lambda a: np.asarray(a)[m]
        scale = np.maximum(np.abs(s(c["gap_log"])), 1.0) * TOL
        out.append(dict(operator=name, year=y, n_dates=int(m.sum()), meaning=meaning,
                        viol_log=float((s(c["gap_log"]) < -scale).mean()), viol_arith=float((s(c["gap_ar"]) < -TOL).mean()),
                        mean_gap_log=float(s(c["gap_log"]).mean()), mean_gap_arith=float(s(c["gap_ar"]).mean()),
                        bias_geo_over_arith=float(s(c["F_geo"]).mean() / s(c["F_ar"]).mean()), mean_rv_over_F_geo=float(np.mean(s(rv)) / s(c["F_geo"]).mean()),
                        mean_rv_over_F_arith=float(np.mean(s(rv)) / s(c["F_ar"]).mean()), qlike_geo=float(qlike(s(rv), s(c["F_geo"])).mean()), qlike_arith=float(qlike(s(rv), s(c["F_ar"])).mean()),
                        resid_vs_saved=float(np.nan if F_saved is None else np.max(np.abs(s(c["F_ar"] if kind == "arith" else c["F_geo"]) / s(F_saved) - 1))), conclusion=""))
    return out


def load_ops() -> list:
    """-> [(name, meaning, dates, rv, Fk, pi, F_saved)] для X1, MoE V7 cfg6, X2, NTP2, K8 на общих датах X1."""
    from models_v2.s15 import de0_data as DD
    from models_v2.s14 import v7_0_io as IO
    R = DD.IO.ROOT
    x1, x2 = pd.read_parquet(R / "data/regen/harj_deep/E1/x1.parquet"), pd.read_parquet(R / "data/regen/harj_deep/E2/x2.parquet")
    c6, c8 = pd.read_parquet(IO.OUT / "2" / "cfg6.parquet"), pd.read_parquet(R / "data/regen/harj_hybrid_v4/C8.parquet")
    d = DD.load()
    idx = pd.DatetimeIndex(x1["d"])
    for t in (x2, c6):
        if not np.array_equal(pd.DatetimeIndex(t["d"]), idx):
            raise SystemExit("даты X2 / cfg6 ≠ датам X1")
    c8 = c8[c8["ticker"] == "SBER"].assign(date=lambda t: pd.to_datetime(t["date"])).set_index("date")["F"].reindex(idx).to_numpy(float)
    M, rv, f = x1["M"].to_numpy(float), x1["rv"].to_numpy(float), d["f"]
    p1, pn = x1["pi"].to_numpy(float), None
    ntp = f["NTP2"]
    pn = np.log(ntp / M) / np.log(c8 / M)                       # NTP2: F = M^(1−π)·C8^π ⇒ π восстановлен из сохранённого ряда (проверка: π ∈ [0, 1])
    H = c6["H"].to_numpy(float)
    Fm = np.stack([H, H * np.exp(c6["g1"].to_numpy(float)), H * np.exp(c6["g2"].to_numpy(float))], 1)
    pm = c6[["pi0", "pi1", "pi2"]].to_numpy(float)
    pm_ok = (c6["fitted"].to_numpy(float) > 0)                  # до первой обученной головы G = 0 (π нет)
    h2, p2 = x2["H"].to_numpy(float), x2["p"].to_numpy(float)
    X2 = np.stack([h2 * x2["cn"].to_numpy(float), h2 * x2["cs"].to_numpy(float)], 1)
    K = np.stack([f["K1"], f["K2"], f["K3"]], 1)
    ops = [("X1", "геометрический|эксперты M, C — прогнозы условного среднего RV (уровни)", idx, rv, np.stack([M, x1["C"].to_numpy(float)], 1), np.stack([1 - p1, p1], 1), x1["F"].to_numpy(float)),
           ("NTP2", "геометрический|эксперты M, C8 — прогнозы условного среднего RV (уровни)", idx, rv, np.stack([M, c8], 1), np.stack([1 - pn, pn], 1), ntp),
           ("K8", "геометрический|эксперты K1, K2, K3 (веса ½, ¼, ¼) — прогнозы условного среднего", idx, rv, K, np.tile([0.5, 0.25, 0.25], (len(idx), 1)), np.asarray(d["all_series"]["K8"], float))]
    ops.append(("V7 MoE cfg6", "геометрический (в логе)|эксперты — лог-поправки g_k к H; F_k = H·e^{g_k}", idx[pm_ok], rv[pm_ok], Fm[pm_ok], pm[pm_ok], c6["F"].to_numpy(float)[pm_ok]))
    ops.append(("X2", "арифметический по множителям|эксперты — мультипликативные поправки c_n, c_s к H", idx, rv, X2, np.stack([1 - p2, p2], 1), x2["F"].to_numpy(float)))
    ops.append(("NTP2_pi_in_01", "служебная|доля дат с восстановленным π ∈ [0,1]", idx, rv, np.stack([M, c8], 1), np.stack([1 - np.clip(pn, 0, 1), np.clip(pn, 0, 1)], 1), np.where((pn >= 0) & (pn <= 1), 1.0, np.nan)))
    return ops


_GEO = ("Йенсен в ln F выполнен на всех датах (QLIKE выпукла по ln F): геометрический оператор не хуже взвешенных потерь экспертов. Но эксперты — прогнозы УСЛОВНОГО СРЕДНЕГО RV (уровни), "
        "а геометрическое среднее средних по AM–GM ниже арифметического (колонка bias_geo_over_arith < 1): оператор соответствует смыслу «лог-ансамбль под QLIKE», не «смесь условных средних».")
CONCL = {"X1": _GEO, "NTP2": _GEO, "K8": _GEO + " Веса фиксированы (½, ¼, ¼).",
         "V7 MoE cfg6": ("Эксперты g_k — лог-поправки к H, оператор Σπ_k g_k аддитивен в логе: соответствует смыслу экспертов; Йенсен в ln F выполнен на всех датах. Смещение к арифметике (колонка bias_geo_over_arith) "
                         "касается только чтения F_k как уровней; resid_vs_saved ≠ 0: сохранённые π и g_k усреднены по seeds раздельно, а F — среднее G по seeds (произведение средних ≠ среднему произведений)."),
         "X2": ("Эксперты c_n, c_s — мультипликативные поправки (уровни множителей), оператор арифметический (1−p)c_n + p·c_s. Йенсен в ln F выполнен на всех датах; арифметический по F нарушается на заметной доле дат "
                "(потеря по F выпукла только при F < 2·RV): арифметический оператор НЕ наследует гарантию «не хуже экспертов». resid_vs_saved ≠ 0: сохранены средние по seeds p, c_n, c_s, а F — среднее ln m по seeds."),
         "NTP2_pi_in_01": "Служебная строка: π NTP2 восстановлен из сохранённых F, M, C8; resid_vs_saved = доля дат, где π ∈ [0, 1] (1 = восстановление согласовано с формулой M^(1−π)·C8^π)."}


def mix_table() -> pd.DataFrame:
    df = _mix_table()
    df["conclusion"] = df["operator"].map(CONCL)
    return df


def _mix_table() -> pd.DataFrame:
    rows = []
    for name, meaning, dates, rv, Fk, pi, Fs in load_ops():
        if name == "NTP2_pi_in_01":
            rows.append(dict(operator=name, year="all", n_dates=len(dates), meaning=meaning, viol_log=np.nan, viol_arith=np.nan, mean_gap_log=np.nan, mean_gap_arith=np.nan, bias_geo_over_arith=np.nan,
                             mean_rv_over_F_geo=np.nan, mean_rv_over_F_arith=np.nan, qlike_geo=np.nan, qlike_arith=np.nan, resid_vs_saved=float(np.isfinite(Fs).mean()), conclusion=""))
            continue
        rows += rows_for(name, meaning, dates, rv, Fk, pi, Fs)
    return pd.DataFrame(rows)
