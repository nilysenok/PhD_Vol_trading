"""V5, лист 1b, G16: функции φ₁…φ₆ (план, п. 6.3), известные на конец дня d до прогноза, и стандартизация только по расширяющемуся прошлому бумаги (дни < d; первые 60 наблюдений — пропуск).
Диагностика задним числом; путём прогноза не импортируется (тест AST). Определения RV, BV, J не меняются: берутся из канонических дневных таблиц.
Уточнения (объявлены): φ₁ — ln RV_d минус среднее ln RV за 252 дня до d; φ₂ — ln RV_d минус ln среднего RV за 5 дней до d; φ₃ — EWMA (полураспад 5, включая d) от J/RV;
φ₄ — среднее s(j,d) по j ≠ i к K8 по парам, закрытым к d; φ₅ — |ln(M/K1)| прогнозов строки d; φ₆ — ln sd (20 закрытых пар до d включительно) от ln(RV/K8)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s12 import v5_0_dx_score as DS

MIN_OBS = 60
PHI = ["phi1", "phi2", "phi3", "phi4", "phi5", "phi6"]
RAW = ["x%d" % i for i in range(1, 7)]


def expanding_z(x: pd.Series, min_obs: int = MIN_OBS, whole: bool = False) -> pd.Series:
    """z-оценка по среднему и sd предыдущих наблюдений (сдвиг на 1). `whole=True` — ТОЛЬКО мутация теста: статистики по всей выборке."""
    if whole:
        mu, sd = x.mean(), x.std()
        return ((x - mu) / sd).where(np.arange(len(x)) >= min_obs)
    p = x.shift(1).expanding(min_periods=min_obs)
    sd = p.std()
    return ((x - p.mean()) / sd).where(sd > 0)


def daily_raw(b: pd.DataFrame) -> pd.DataFrame:
    """b: (ticker, date, rv, j) -> x1, x2, x3 на канонических днях бумаги."""
    d = b.dropna(subset=["rv"]).sort_values(["ticker", "date"]).reset_index(drop=True)[["ticker", "date", "rv", "j"]]
    g = d.groupby("ticker", sort=False)
    lr = np.log(d["rv"])
    d["x1"] = lr - lr.groupby(d["ticker"]).transform(lambda s: s.shift(1).rolling(252, min_periods=252).mean())
    prev5 = g["rv"].transform(lambda s: s.shift(1).rolling(5, min_periods=5).mean())
    d["x2"] = lr - np.log(prev5)
    d["x3"] = (d["j"] / d["rv"]).groupby(d["ticker"]).transform(lambda s: s.ewm(halflife=5, adjust=False).mean())
    return d[["ticker", "date", "x1", "x2", "x3"]]


def frame_raw(c: pd.DataFrame) -> pd.DataFrame:
    """c: общий кадр M16 (контроли, K8, M, rv_tplus, t_plus) -> x4, x5, x6 по (бумага, d)."""
    w = DS.known_scores(c, "K8").sort_values(["ticker", "date"]).reset_index(drop=True)
    sd = w.groupby("ticker")["ls_own"].transform(lambda s: s.rolling(20, min_periods=20).std())
    out = w[["ticker", "date"]].copy()
    out["x4"] = w["s_oth"].to_numpy()
    out["x5"] = np.abs(np.log(w["M"] / w["k1"])).to_numpy()
    out["x6"] = np.log(sd).to_numpy()
    return out


def phi_table(b: pd.DataFrame, c: pd.DataFrame, whole: bool = False) -> pd.DataFrame:
    """(ticker, date, phi1…phi6): стандартизация по прошлому каждой бумаги на её дневной истории (до 2017 тоже, если ряд есть)."""
    t = daily_raw(b).merge(frame_raw(c), on=["ticker", "date"], how="left").sort_values(["ticker", "date"]).reset_index(drop=True)
    for r, p in zip(RAW, PHI):
        t[p] = t.groupby("ticker", sort=False)[r].transform(lambda s: expanding_z(s, whole=whole))
    return t[["ticker", "date"] + PHI]
