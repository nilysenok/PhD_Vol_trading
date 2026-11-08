"""V4-0, G1/G2: инвентарь сохранённых прогнозов и проверка соответствия цели RV, дат, источников и PIT (OOF). Модели не обучаются."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s11 import v4_io as IO

SHIFTS = (-2, -1, 0, 1)
LAST_KNOWN = -1      # Д1: прогноз строки i известен на закрытии d_i, последний известный RV — цель строки i−1; максимум корреляции обязан быть там


def by_year(dates: pd.Series) -> str:
    return ";".join("%d:%d" % (y, n) for y, n in dates.dt.year.value_counts().sort_index().items())


def inventory_rows(t: str) -> list:
    """Строки INVENTORY: бумага × {K1, K2, K3, LGBM-D2} + общие даты H∩K2∩K3."""
    srcs = {"K1": (IO.load_h(t), "h", "H"), "K2": (IO.load_k23(t), "k2", "K2"), "K3": (IO.load_k23(t), "k3", "K3"), "LGBM-D2": (IO.load_d2(t), "d2", "D2")}
    rows = []
    for name, (f, col, key) in srcs.items():
        g = f[np.isfinite(f[col]) & (f[col] > 0)]
        p = IO.path_of(key, t)
        rows.append({"ticker": t, "model": name, "file": str(p.relative_to(IO.P.ROOT)), "sha256": IO.sha256(p), "n": len(g), "first_date": g["date"].min().date(), "last_date": g["date"].max().date(),
                     "first_target": g["t_plus"].min().date(), "last_target": g["t_plus"].max().date(), "by_year_target": by_year(g["t_plus"])})
    a = IO.aligned(t)
    rows.append({"ticker": t, "model": "H∩K2∩K3", "file": "—", "sha256": "—", "n": len(a), "first_date": a["date"].min().date(), "last_date": a["date"].max().date(),
                 "first_target": a["t_plus"].min().date(), "last_target": a["t_plus"].max().date(), "by_year_target": by_year(a["t_plus"])})
    return rows


def shift_profile(rv, f) -> dict:
    """corr(ln f_i, ln rv_{i+j}) при j = −2…+1 (Д1). rv — цель строки; honest-прогноз строки i максимально связан с последним известным RV (j = −1)."""
    lf, lr = np.log(np.asarray(f, float)), np.log(np.asarray(rv, float))
    ok = np.isfinite(lf)
    out, n = {}, len(lf)
    for j in SHIFTS:
        i = np.arange(max(0, -j), min(n, n - j))
        i = i[ok[i] & np.isfinite(lr[i + j])]
        out[j] = float(np.corrcoef(lf[i], lr[i + j])[0, 1]) if len(i) > 30 else float("nan")
    return out


def shift_ok(rv, f) -> bool:
    p = shift_profile(rv, f)
    return bool(all(np.isfinite(v) for v in p.values()) and max(p, key=p.get) == LAST_KNOWN)


def _eq(a, b) -> int:
    """Число побитно разных элементов (NaN = NaN)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())


def rv_vs_daily(f: pd.DataFrame, daily: pd.DataFrame) -> int:
    ref = daily["rv"].reindex(pd.DatetimeIndex(f["t_plus"])).to_numpy(float)
    return int((~np.isclose(f["rv"].to_numpy(float), ref, rtol=1e-12, atol=0)).sum())


def tplus_not_next(f: pd.DataFrame, daily: pd.DataFrame) -> int:
    """Строки, где t⁺ не следующий собственный торговый день бумаги после d."""
    idx = daily.index.to_numpy("datetime64[ns]")
    pos = np.searchsorted(idx, f["date"].to_numpy("datetime64[ns]"), side="right")
    nxt = np.where(pos < len(idx), idx[np.minimum(pos, len(idx) - 1)], np.datetime64("NaT"))
    return int((nxt != f["t_plus"].to_numpy("datetime64[ns]")).sum())


def oof_meta(t: str, k: pd.DataFrame) -> dict:
    """Метаданные K2/K3: годовые HP (И2) покрывают каждый год прогнозов; hp_year = год цели t⁺ (HP года Y обучены на парах с целями < 01.01.Y: `select.cut_row`); первый год прогноза ≥ y0_used."""
    m = json.loads((IO.PAN / ("k23_meta_%s.json" % t)).read_text())
    yrs = {int(y) for y in json.loads(m["i2"])}
    f = k[np.isfinite(k["k2"]) | np.isfinite(k["k3"])]
    fy = f["t_plus"].dt.year.to_numpy()
    return {"y0_used": m["y0_used"], "n_forecast_meta": m["n_forecast"], "n_forecast_file": len(f), "hp_year_ok": bool((f["hp_year"].to_numpy() == fy).all()),
            "i2_covers_years": bool(set(np.unique(fy).tolist()) <= yrs), "first_year_ok": bool(len(f) and fy.min() >= m["y0_used"])}


def audit_align(t: str) -> dict:
    """Строка AUDIT_ALIGN бумаги. Все ряды берутся через `v4_io` (боевой путь); независимая опора RV цели — daily_<T>.rv."""
    h, k, d2, eng, daily = IO.load_h(t), IO.load_k23(t), IO.load_d2(t), IO.load_eng(t), IO.load_daily(t)
    a = IO.aligned(t)
    mk = h.merge(k, on="date", suffixes=("", "_k"))
    r = {"ticker": t, "n_h": len(h), "n_k23": len(k), "n_aligned": len(a), "dates_equal_h_k23": bool(len(h) == len(k) and (h["date"].to_numpy() == k["date"].to_numpy()).all()),
         "t_plus_diff_h_k23": int((mk["t_plus"] != mk["t_plus_k"]).sum()), "rv_diff_h_k23": _eq(mk["rv"], mk["rv_k"])}
    for name, f in (("h", h), ("k23", k), ("eng", eng), ("d2", d2)):
        r["rv_vs_daily_" + name] = rv_vs_daily(f, daily)
        r["tplus_not_next_" + name] = tplus_not_next(f, daily)
    for name, f in (("k23", k), ("eng", eng), ("d2", d2)):
        mm = h.merge(f[["date", "k1"]], on="date")
        r["k1_diff_" + name] = _eq(mm["h"], mm["k1"]) + (len(h) - len(mm))
    for col in ("h", "k2", "k3"):
        r["shift_ok_" + col] = shift_ok(a["rv"], a[col])
    ad = a[np.isfinite(a["d2"])]
    r["shift_ok_d2"] = shift_ok(ad["rv"], ad["d2"])
    r.update(oof_meta(t, k))
    r["oof_k2k3"] = bool(r["hp_year_ok"] and r["i2_covers_years"] and r["first_year_ok"] and r["n_forecast_meta"] == r["n_forecast_file"])
    r["align_ok"] = bool(r["dates_equal_h_k23"] and r["t_plus_diff_h_k23"] == 0 and r["rv_diff_h_k23"] == 0 and all(r[c] == 0 for c in r if c.startswith(("rv_vs_daily_", "tplus_not_next_", "k1_diff_")))
                         and all(r["shift_ok_" + c] for c in ("h", "k2", "k3", "d2")))
    r["status"] = "OOF-пригоден" if r["align_ok"] and r["oof_k2k3"] else "НЕ пригоден"
    return r


def audit_all(tickers=None) -> pd.DataFrame:
    return pd.DataFrame([audit_align(t) for t in (tickers or IO.T17)])
