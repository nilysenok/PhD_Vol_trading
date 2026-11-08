"""V4-0: чтение сохранённых прогнозов (K1 = HAR-J, K2 = XGB, K3 = LGBM, LGBM-D2) и RV цели; запись выходов листа. Ничего не обучает, `models_v2/s10` — только константы путей."""
from __future__ import annotations

import functools
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s10 import hp_paths as P

PAN = P.OUT                                   # data/regen/harj_panel_hybrid (только чтение)
REP = P.ROOT / "docs/otchety/harj_hybrid_v4"
T17 = list(P.MAIN17)
M16 = list(P.M16)
STATUS = "DEVELOPMENT; HISTORICAL WALK-FORWARD 2017–02.2026, не независимый OOS; Ж-222 лист V4-0/A; первая цифра — потолок"
WINDOWS = {"2017_2021": (2017, 2021), "2022_2025": (2022, 2025), "2026": (2026, 2026), "all": (2017, 2026)}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def path_of(model: str, t: str) -> Path:
    return {"H": PAN / ("har_%s.parquet" % t), "K2": PAN / ("k23_%s.parquet" % t), "K3": PAN / ("k23_%s.parquet" % t), "D2": PAN / "hyb_lgbm_p17.parquet",
            "ENG": PAN / "engine_panel_p17.parquet", "DAILY": PAN / ("daily_%s.parquet" % t)}[model]


def load_h(t: str) -> pd.DataFrame:
    """K1: date (d), t_plus, rv (цель), h (σ̂²)."""
    f = pd.read_parquet(path_of("H", t))
    return pd.DataFrame({"date": pd.to_datetime(f["d"]), "t_plus": pd.to_datetime(f["target_date"]), "rv": f["rv"].to_numpy(float), "h": f["sigma2"].to_numpy(float)})


def load_k23(t: str) -> pd.DataFrame:
    """date, t_plus, rv, k1 (копия K1 в файле K2/K3), k2, k3, hp_year."""
    f = pd.read_parquet(path_of("K2", t))
    return pd.DataFrame({"date": pd.to_datetime(f["date"]), "t_plus": pd.to_datetime(f["t_plus"]), "rv": f["rv_tplus"].to_numpy(float), "k1": f["sigma2_k1"].to_numpy(float),
                         "k2": f["sigma2_k2"].to_numpy(float), "k3": f["sigma2_k3"].to_numpy(float), "hp_year": f["hp_year"].to_numpy(float)})


@functools.lru_cache(maxsize=1)
def _d2_all() -> pd.DataFrame:
    return pd.read_parquet(path_of("D2", "x"), columns=["ticker", "date", "t_plus", "rv_tplus", "sigma2_k1", "sigma2_D2"])


def load_d2(t: str) -> pd.DataFrame:
    """LGBM-D2 (контроль): date, t_plus, rv, k1, d2."""
    f = _d2_all()
    f = f[f["ticker"] == t]
    return pd.DataFrame({"date": pd.to_datetime(f["date"]).to_numpy(), "t_plus": pd.to_datetime(f["t_plus"]).to_numpy(), "rv": f["rv_tplus"].to_numpy(float), "k1": f["sigma2_k1"].to_numpy(float),
                         "d2": f["sigma2_D2"].to_numpy(float)})


@functools.lru_cache(maxsize=1)
def _eng_all() -> pd.DataFrame:
    return pd.read_parquet(path_of("ENG", "x"), columns=["ticker", "date", "t_plus", "rv_tplus", "sigma2_k1"])


def load_eng(t: str) -> pd.DataFrame:
    f = _eng_all()
    f = f[f["ticker"] == t]
    return pd.DataFrame({"date": pd.to_datetime(f["date"]).to_numpy(), "t_plus": pd.to_datetime(f["t_plus"]).to_numpy(), "rv": f["rv_tplus"].to_numpy(float), "k1": f["sigma2_k1"].to_numpy(float)})


def load_daily(t: str) -> pd.DataFrame:
    """Дневной ряд бумаги (независимый источник RV цели), индекс — дата."""
    f = pd.read_parquet(path_of("DAILY", t))
    f.index = pd.to_datetime(f.index)
    return f


def aligned(t: str) -> pd.DataFrame:
    """Общие даты H ∩ K2 ∩ K3 (все σ̂² конечны и > 0) бумаги + D2 (NaN там, где его нет). Колонки: date, t_plus, rv, h, k2, k3, d2. Боевой путь выравнивания для A и аудита."""
    h, k = load_h(t), load_k23(t)
    m = h.merge(k[["date", "t_plus", "rv", "k2", "k3"]], on="date", suffixes=("", "_k"))
    ok = (m["t_plus"] == m["t_plus_k"]) & (m["rv"] == m["rv_k"])
    for c in ("h", "k2", "k3", "rv"):
        ok &= np.isfinite(m[c]) & (m[c] > 0)
    m = m[ok].drop(columns=["t_plus_k", "rv_k"])
    d2 = load_d2(t)[["date", "d2"]]
    m = m.merge(d2, on="date", how="left")
    return m.sort_values("date").reset_index(drop=True)


def write_csv(df: pd.DataFrame, name: str, note: str) -> Path:
    REP.mkdir(parents=True, exist_ok=True)
    p = REP / name
    with open(p, "w", encoding="utf-8") as f:
        f.write("# %s; %s\n" % (STATUS, note))
        df.to_csv(f, index=False, lineterminator="\n")
    return p


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(REP / name, comment="#")
