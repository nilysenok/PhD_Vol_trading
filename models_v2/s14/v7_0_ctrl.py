"""V7-0, G1: замороженные контроли K1, K2, K3, V4:NTP2, V6:A2, V6:E2. Читаются СОХРАНЁННЫЕ прогнозы; ничего не обучается и не пишется. Выравнивание — как `v4_io.aligned`: общие даты H ∩ K2 ∩ K3 (σ̂² конечны и > 0).
Проверки: sha256 файла = `CMP_SERIES.csv`; sha256 массива прогноза на общих датах = `CMP_SERIES.csv`; QLIKE (определение `models_v2.metrics.qlike`) по окнам года цели t⁺ = `CMP_WINDOWS.csv` (допуск 1e-12)."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s14 import v7_0_io as IO

SERIES = ["K1", "K2", "K3", "V4:NTP2", "V6:A2", "V6:E2"]
WIN = {"2017_2021": (2017, 2021), "2022_2025": (2022, 2025), "2026": (2026, 2026), "all": (2017, 2026)}
TOL_Q = 1e-12


def sha_arr(x) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x, float)).tobytes()).hexdigest()


def aligned() -> pd.DataFrame:
    """Индекс — дата d; колонки t_plus, rv, K1, K2, K3 на общих датах H ∩ K2 ∩ K3."""
    h = pd.read_parquet(IO.PAN / "har_SBER.parquet")
    k = pd.read_parquet(IO.PAN / "k23_SBER.parquet")
    h = pd.DataFrame({"date": pd.to_datetime(h["d"]), "t_plus": pd.to_datetime(h["target_date"]), "rv": h["rv"].to_numpy(float), "K1": h["sigma2"].to_numpy(float)})
    k = pd.DataFrame({"date": pd.to_datetime(k["date"]), "t_plus_k": pd.to_datetime(k["t_plus"]), "rv_k": k["rv_tplus"].to_numpy(float), "K2": k["sigma2_k2"].to_numpy(float), "K3": k["sigma2_k3"].to_numpy(float)})
    m = h.merge(k, on="date")
    ok = (m["t_plus"] == m["t_plus_k"]) & (m["rv"] == m["rv_k"])
    for c in ("K1", "K2", "K3", "rv"):
        ok &= np.isfinite(m[c]) & (m[c] > 0)
    return m[ok].drop(columns=["t_plus_k", "rv_k"]).sort_values("date").set_index("date")


def forecast(name: str, path, idx: pd.DatetimeIndex, w: pd.DataFrame) -> np.ndarray:
    if name in ("K1", "K2", "K3"):
        return w[name].to_numpy(float)
    f = pd.read_parquet(path)
    if "ticker" in f.columns:
        f = f[f["ticker"] == "SBER"]
    d = pd.to_datetime(f["d"] if "d" in f.columns else f["date"])
    return pd.Series(f["F"].to_numpy(float), index=pd.DatetimeIndex(d)).reindex(idx).to_numpy(float)


def wmask(tp, win: str) -> np.ndarray:
    y = pd.DatetimeIndex(tp).year.to_numpy()
    lo, hi = WIN[win]
    return (y >= lo) & (y <= hi)


def frozen() -> pd.DataFrame:
    """Что объявлено в `CMP_SERIES.csv` для шести контролей (первая строка ряда)."""
    c = pd.read_csv(IO.CMP / "CMP_SERIES.csv", comment="#")
    return c[c["series"].isin(SERIES)].drop_duplicates("series").set_index("series").loc[SERIES]


def check_rows() -> pd.DataFrame:
    """Строка на (ряд, проверка): sha файла, sha прогноза, QLIKE по четырём окнам. status — «совпало» / «расхождение»."""
    w, fz = aligned(), frozen()
    cw = pd.read_csv(IO.CMP / "CMP_WINDOWS.csv", comment="#").set_index(["series", "window"])
    rows = []
    for s in SERIES:
        path = IO.ROOT / fz.loc[s, "file"]
        f = forecast(s, path, w.index, w)
        for kind, got, want in (("sha256_file", IO.sha_file(path), fz.loc[s, "sha256_file"]), ("sha256_forecast", sha_arr(f), fz.loc[s, "sha256_forecast"]), ("n_dates", len(f), int(fz.loc[s, "n_dates"]))):
            rows.append(dict(series=s, check=kind, window="-", got=str(got), expected=str(want), abs_diff=np.nan, status="совпало" if str(got) == str(want) else "расхождение"))
        q = metrics.qlike(w["rv"].to_numpy(float), f)
        for win in WIN:
            m = wmask(w["t_plus"], win)
            v, ref = float(q[m].mean()), float(cw.loc[(s, win), "qlike"])
            n_ok = int(m.sum()) == int(cw.loc[(s, win), "n"])
            rows.append(dict(series=s, check="qlike", window=win, got=repr(v), expected=repr(ref), abs_diff=abs(v - ref), status="совпало" if abs(v - ref) <= TOL_Q and n_ok else "расхождение"))
    return pd.DataFrame(rows)
