"""V4-C, G15: построение Z_<T>.parquet по 17 бумагам из сохранённых входов (только чтение), проверка сдвига (Д1) по каждому признаку, проверка обрезкой, Z_REPORT.csv."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s10 import hp_daily
from models_v2.s11 import v4_audit as AU, v4_b_data as D, v4_c_cache as CC, v4_io as IO, v4_z_feat as ZF, v4_z_shape as SH

SHIFTS = (-2, -1, 0, 1)
LAST_KNOWN = -1
LEAK_MARGIN = 0.05            # утечка: |corr| при сдвиге 0/+1 превышает |corr| при −1 больше, чем на 0,05
DEV_END = pd.Timestamp("2021-12-31")
TRUNC_DATES = ("2018-06-15", "2020-03-12", "2021-09-10")
SOURCE = {"ell": "K1, K2, K3 (прогнозы дня d)", "abs_ell": "K1, K2, K3 (прогнозы дня d)", "ewqd": "RV цели закрытых пар + K1, K2, K3", "lerr_h": "RV дня d (цель пары с t⁺ = d) + K1 прошлого дня",
          "lerr_m": "RV дня d (цель пары с t⁺ = d) + K2, K3 прошлого дня", "med_lrv": "daily_<T>.rv (M10), медиана по 17 бумагам", "dlrv": "daily_<T>.rv (M10)",
          "jshare": "daily_<T>.j / rv (определение J — как в K1, daily.build_daily)", "sur_mkt": "медиана по 17 бумагам ln(RV_d / K1_{d−1})",
          "lasth": "daily_<T>.rv_close / rv (последний час, Ж-214)", "herf": "10-минутные бары bars_10m_full, hv2_m10.bar_returns"}


def real_inputs() -> dict:
    return {t: {"daily": IO.load_daily(t), "herf": SH.herf_by_day(hp_daily.read_bars(t))} for t in IO.T17}


def build_all() -> pd.DataFrame:
    """Длинная таблица Z по 17 бумагам на общих датах H∩K2∩K3 (боевой путь)."""
    return ZF.build_panel_z(D.panel(), real_inputs(), IO.T17)


def z_by_ticker(key: str) -> pd.DataFrame:
    memo = {}

    def mk(t):
        def build():
            if "z" not in memo:
                memo["z"] = build_all()
            return {"": memo["z"][memo["z"]["ticker"] == t].reset_index(drop=True)}
        return build
    return pd.concat([CC.cached("Z", "Z_" + t, key, mk(t))[""] for t in IO.T17], ignore_index=True)


def profile(x: np.ndarray, lrv_target: np.ndarray) -> dict:
    """corr(x_i, ln rv_{i+j}) при j = −2…+1 (Д1): максимум |corr| должен быть на последнем известном дне (j = −1)."""
    out, n = {}, len(x)
    for j in SHIFTS:
        i = np.arange(max(0, -j), min(n, n - j))
        i = i[np.isfinite(x[i]) & np.isfinite(lrv_target[i + j])]
        out[j] = float(np.corrcoef(x[i], lrv_target[i + j])[0, 1]) if len(i) > 30 and np.std(x[i]) > 0 else float("nan")
    return out


def leak(p: dict) -> bool:
    """Подозрение на утечку: сильнее всего связь на сдвиге 0 или +1 (больше −1 на LEAK_MARGIN)."""
    a = {j: abs(v) for j, v in p.items() if np.isfinite(v)}
    return bool(a and max(a, key=a.get) in (0, 1) and max(a[0], a.get(1, 0)) - a.get(LAST_KNOWN, 0) > LEAK_MARGIN)


def shift_table(Z: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    """Профиль сдвига каждого признака (среднее по бумагам per-ticker корреляций), цели — до 2021."""
    m = Z.merge(P[["ticker", "date", "rv"]], on=["ticker", "date"])
    m = m[m["t_plus"] <= DEV_END]
    rows = []
    rel = related_levels_ok(m.merge(P[["ticker", "date", "h", "k2", "k3"]], on=["ticker", "date"]))
    for f in ZF.ALL:
        ps = [profile(g[f].to_numpy(float), np.log(g["rv"].to_numpy(float))) for _, g in m.sort_values("date").groupby("ticker")]
        mean = {j: float(np.nanmean([p[j] for p in ps])) for j in SHIFTS}
        via_levels = f in ("ell", "abs_ell")       # разность прогнозов не имеет персистентности уровня: профиль не обязан иметь максимум на −1 — вторая проверка Д1 через связанные величины (ln H, ln K2, ln K3)
        rows.append({"feature": f, **{"corr_shift_%+d" % j: v for j, v in mean.items()}, "peak_shift": max(mean, key=lambda j: abs(mean[j])), "check": "levels H, K2, K3" if via_levels else "profile",
                     "leak_suspect": (not rel) if via_levels else leak(mean), "flat_profile": bool(max(abs(v) for v in mean.values()) < 0.05)})
    return pd.DataFrame(rows)


def related_levels_ok(m: pd.DataFrame) -> bool:
    """Д1 для разности прогнозов: уровни ln H, ln K2, ln K3 каждой бумаги имеют максимум профиля на последнем известном дне (j = −1) — `v4_audit.shift_ok`."""
    return all(AU.shift_ok(g["rv"].to_numpy(float), g[c].to_numpy(float)) for _, g in m.sort_values("date").groupby("ticker") for c in ("h", "k2", "k3"))


def truncation_ok(a: pd.DataFrame, daily: pd.DataFrame, herf: pd.Series, day) -> bool:
    """Подмена данных после дня d не меняет собственные признаки строки d (обрезка + цель строки d обнулена)."""
    day = pd.Timestamp(day)
    full = ZF.own(a, daily, herf)
    k = a[a["date"] <= day].copy()
    k.loc[k.index[-1], ["rv"]] = np.nan                      # цель строки d (RV d+1) неизвестна
    cut = ZF.own(k, daily[daily.index <= day], herf[herf.index <= day])
    i = len(k) - 1                                           # последняя строка с date <= d — одна и та же позиция в full и в cut
    f, c = full.iloc[i].to_numpy(float), cut.iloc[i].to_numpy(float)
    return bool(np.allclose(f, c, rtol=0, atol=1e-12, equal_nan=True))


def report(Z: pd.DataFrame, P: pd.DataFrame, inputs: dict, key: dict) -> pd.DataFrame:
    sh = shift_table(Z, P)
    tr = {t: [truncation_ok(P[P["ticker"] == t].reset_index(drop=True), inputs[t]["daily"], inputs[t]["herf"], d) for d in TRUNC_DATES] for t in IO.T17}
    ok_tr = all(all(v) for v in tr.values())
    rows = []
    for f in ZF.ALL:
        r = sh[sh["feature"] == f].iloc[0].to_dict()
        rows.append({**r, "source": SOURCE[f], "n": len(Z), "n_nan": int(Z[f].isna().sum()), "coverage": 1.0 - float(Z[f].isna().mean()), "truncation_ok": ok_tr,
                     "cross_tickers": "17 (MAIN17)" if f in ZF.CROSS else "—", "inputs_sha": key["inputs"][:16], "code_sha": key["code"][:16]})
    return pd.DataFrame(rows)
