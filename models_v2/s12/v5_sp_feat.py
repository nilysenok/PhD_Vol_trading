"""V5 лист 2 (Ж-226), Spectral Energy: признаки дня d из энергий полос DCT-II (low/mid/high, кэш листа 0 `bands_*.parquet`) и формы дня H3 V4 (lasth, herf, firsth). Только прошлое: значение строки d известно на конец дня d.
Доли q_b = E_b / ΣE (Σ_b E_b = RV, тождество листа 0); π_b — расширяющееся среднее q_b по дням ≤ d (до 60 дней — 1/3); среднее q_b за 5 и 22 строки бумаги (полные окна).
z-стандартизация формы — по расширяющемуся прошлому бумаги (до 60 дней — 0). Лог доли ограничен снизу 1e-6 (защита ln 0; в данных не срабатывает).
Ось — даты d (объединение дней бумаг); бумаги — только список `tickers` (17; E11 не входит), межбумажный фактор SP3 — по множеству `keep` (в LOTO без исключённой)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BANDS = ("low", "mid", "high")
SHAPE = ("lasth", "herf", "firsth")
QF = 1e-6
MIN_PI = 60
MIN_Z = 60
LAGS = ("d", "m5", "m22")
LAG_NAMES = {"SP2": list(LAGS), "SP3": list(LAGS) + ["x"], "SP4": list(LAGS) + list(SHAPE)}


@dataclass
class Feat:
    tick: list
    D: pd.DatetimeIndex
    E: np.ndarray        # (nD, m, 3) энергии полос дня (1e4·Σc²)
    RV: np.ndarray       # (nD, m) RV канона дня
    E5: np.ndarray
    E22: np.ndarray
    R5: np.ndarray
    R22: np.ndarray
    q: np.ndarray        # (nD, m, 3) доли
    q5: np.ndarray
    q22: np.ndarray
    pi: np.ndarray       # (nD, m, 3)
    shape: np.ndarray    # (nD, m, 3) стандартизованная форма (NaN до 60 дней)


def _one(g: pd.DataFrame, fh: pd.Series) -> dict:
    """Серии одной бумаги по её собственным дням (строки отсортированы по дате)."""
    e = g[["e_" + b for b in BANDS]].to_numpy(float)
    tot = e.sum(axis=1)
    q = pd.DataFrame(e / tot[:, None], index=g["date"])
    E = pd.DataFrame(e, index=g["date"])
    rv = pd.Series(g["rv"].to_numpy(float), index=g["date"])
    cnt = np.arange(1, len(g) + 1)[:, None]
    pi = np.where(cnt >= MIN_PI, q.expanding().mean().to_numpy(), 1.0 / 3.0)
    raw = pd.DataFrame({"lasth": (g["rv_close"] / g["rv"]).where(g["rv"] > 0).to_numpy(float), "herf": g["herf"].to_numpy(float), "firsth": fh.reindex(g["date"]).to_numpy(float)}, index=g["date"])
    mu, sd = raw.expanding(min_periods=MIN_Z).mean(), raw.expanding(min_periods=MIN_Z).std()
    z = ((raw - mu) / sd.where(sd > 0)).to_numpy(float)
    return dict(E=E.to_numpy(), RV=rv.to_numpy(), E5=E.rolling(5, min_periods=5).mean().to_numpy(), E22=E.rolling(22, min_periods=22).mean().to_numpy(), R5=rv.rolling(5, min_periods=5).mean().to_numpy(), R22=rv.rolling(22, min_periods=22).mean().to_numpy(),
                q=q.to_numpy(), q5=q.rolling(5, min_periods=5).mean().to_numpy(), q22=q.rolling(22, min_periods=22).mean().to_numpy(), pi=pi, shape=z, date=g["date"].to_numpy())


def build(bands: pd.DataFrame, firsth: pd.DataFrame, tickers: list) -> Feat:
    """bands — кэш листа 0 (ticker, date, e_low, e_mid, e_high, rv, rv_close, herf, …); firsth — (ticker, date, firsth). Дни без RV канона не входят."""
    tickers = list(tickers)
    b = bands[bands["ticker"].isin(tickers)].dropna(subset=["rv", "e_low", "e_mid", "e_high"])
    D = pd.DatetimeIndex(sorted(b["date"].unique()))
    m = len(tickers)
    shp = {"E": 3, "E5": 3, "E22": 3, "q": 3, "q5": 3, "q22": 3, "pi": 3, "shape": 3, "RV": 0, "R5": 0, "R22": 0}
    out = {k: np.full((len(D), m, v) if v else (len(D), m), np.nan) for k, v in shp.items()}
    for j, t in enumerate(tickers):
        g = b[b["ticker"] == t].sort_values("date")
        if not len(g):
            continue
        fh = firsth[firsth["ticker"] == t].set_index("date")["firsth"]
        r = _one(g, fh)
        ix = D.get_indexer(pd.DatetimeIndex(r["date"]))
        for k in shp:
            out[k][ix, j] = r[k]
    return Feat(tickers, D, **out)


def cross_factor(F: Feat, keep: np.ndarray) -> np.ndarray:
    """Межбумажный фактор SP3 (nD, m, 3): ln среднего q_b бумаг j ≠ i в день d − ln среднего π_b тех же бумаг; j — только из keep (в LOTO исключённая не входит); нет других бумаг — 0."""
    fin = np.isfinite(F.q[:, :, 0]) & np.asarray(keep)[None, :]
    Qz, Pz = np.where(fin[..., None], F.q, 0.0), np.where(fin[..., None], F.pi, 0.0)
    n_oth = fin.sum(axis=1)[:, None] - fin                      # (nD, m): сколько бумаг, кроме своей, есть в день d
    with np.errstate(all="ignore"):
        oq = (Qz.sum(axis=1)[:, None, :] - Qz) / n_oth[..., None]
        op = (Pz.sum(axis=1)[:, None, :] - Pz) / n_oth[..., None]
        f = np.log(np.maximum(oq, QF)) - np.log(np.maximum(op, QF))
    return np.where((n_oth >= 1)[..., None], f, 0.0)


def day_z(F: Feat, cfg: str, keep: np.ndarray) -> np.ndarray:
    """z_b по осям (d, бумага): (nD, m, 3, L); L = 3 (SP2), 4 (SP3), 6 (SP4). Нефинитные компоненты = 0 (нет информации)."""
    lp = np.log(np.maximum(F.pi, QF))
    comp = [np.log(np.maximum(x, QF)) - lp for x in (F.q, F.q5, F.q22)]
    if cfg == "SP3":
        comp.append(cross_factor(F, keep))
    if cfg == "SP4":
        comp += [np.repeat(F.shape[:, :, i:i + 1], 3, axis=2) for i in range(3)]
    return np.nan_to_num(np.stack(comp, axis=-1), nan=0.0, posinf=0.0, neginf=0.0)


def panel_index(P, F: Feat) -> tuple:
    """Для каждой строки панели (τ, бумага) — индекс дня прогноза d на оси F.D и признак «день найден»."""
    dd = np.where(P.pres, P.dd, F.D.to_numpy()[0])
    di = np.minimum(np.searchsorted(F.D.to_numpy(), dd), len(F.D) - 1)
    return di, P.pres & (F.D.to_numpy()[di] == dd)


def panel_z(P, F: Feat, cfg: str, keep: np.ndarray) -> tuple:
    """-> (Z (N, m, 3, L), π (N, m, 3)) на строках панели; строки без найденного дня — z = 0, π = 1/3 (прогноз K8)."""
    di, ok = panel_index(P, F)
    jx = np.broadcast_to(np.arange(len(P.tick))[None, :], di.shape)
    Z = day_z(F, cfg, keep)[di, jx]
    pi = F.pi[di, jx]
    bad = ~ok | ~np.isfinite(pi).all(axis=-1)
    Z[bad], pi[bad] = 0.0, 1.0 / 3.0
    return Z, pi
