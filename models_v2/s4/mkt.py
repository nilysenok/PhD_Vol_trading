"""И3 (Ж-200/Ж-201): рыночные признаки дня d по барам <= 18:50 дня d. День рынка — годный день SBER.

Прокси IMOEX — RV равновзвешенного портфеля 17 бумаг (средняя 10-минутная доходность по бумагам с баром);
IMOEX (ISS) — отдельный источник (путь к parquet с барами), те же правила сессии и исключений О1–О3.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2 import daily

Z99 = 2.3263478740408408   # Φ^{-1}(0,99), α = 0,01
MU43 = 2 ** (2 / 3) * math.gamma(7 / 6) / math.gamma(0.5)
HT_C = math.pi ** 2 / 4 + math.pi - 5
BASE = ("proxy_d", "proxy_w", "proxy_m", "avg_d", "avg_w", "avg_m", "gap_med", "jump_share", "rel_d", "rel_w", "rel_m")
IDX = ("imoex_d", "imoex_w", "imoex_m", "relx_d", "relx_w", "relx_m")


def jump_z(r: pd.Series, day: pd.Series) -> pd.DataFrame:
    """Хуанг–Таучен по дням: z = ((RV−BV)/RV) / sqrt(HT_C/n · max(1, TP/BV²)); r — доходности баров в долях."""
    a = r.abs()
    g = a.groupby(day)
    a1, a2 = g.shift(1), g.shift(2)
    n = r.groupby(day).size().astype(float)
    rv = (r ** 2).groupby(day).sum()
    bv = (math.pi / 2 * (a * a1)).groupby(day).sum()
    tp = n * MU43 ** -3 * n / (n - 2) * ((a ** (4 / 3)) * (a1 ** (4 / 3)) * (a2 ** (4 / 3))).groupby(day).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (rv - bv) / rv / np.sqrt(HT_C / n * np.maximum(1.0, tp / bv ** 2))
    return pd.DataFrame({"z": z.where(bv > 0)})


def paper(bars: pd.DataFrame) -> dict:
    """Одна бумага -> {'day': DataFrame по годным дням (rv, rv_w, rv_m, gap_rel, jump), 'r': Series доходностей баров}."""
    dly, _ = daily.build_daily(bars)
    main = daily.main_session(bars).sort_values("begin").reset_index(drop=True)
    r = daily.intraday_returns(main)
    ok = r["date"].isin(dly.index).to_numpy()
    main, r = main[ok].reset_index(drop=True), r[ok].reset_index(drop=True)
    z = jump_z(r["r"], r["date"])["z"].reindex(dly.index)
    g = main.groupby(main["begin"].dt.normalize())
    gap = (100.0 * np.log(g["open"].first() / g["close"].last().shift(1))) ** 2
    same = pd.Series(dly["seg"].to_numpy() == dly["seg"].shift(1).to_numpy(), index=dly.index)
    day = dly[["rv", "rv_w", "rv_m"]].copy()
    day["gap_rel"] = (gap.reindex(dly.index) / dly["rv_m"]).where(same)
    day["jump"] = (z > Z99).astype(float).where(z.notna())
    return {"day": day, "r": pd.Series(r["r"].to_numpy(), index=pd.DatetimeIndex(main["begin"]))}


def _win(x: np.ndarray, seg: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for s in np.unique(seg):
        m = np.flatnonzero(seg == s)
        out[m] = daily._window_mean(x[m], w)
    return out


def _ln_dwm(rv: np.ndarray, seg: np.ndarray) -> tuple:
    return np.log(rv), np.log(_win(rv, seg, daily.W_WEEK)), np.log(_win(rv, seg, daily.W_MONTH))


def proxy_rv(rs: list, dates: pd.DatetimeIndex) -> np.ndarray:
    """RV (%²) равновзвешенного портфеля по барам со средней по бумагам доходностью; NaN, где баров нет."""
    pr = pd.concat(rs, axis=1).mean(axis=1)
    rv = ((100.0 * pr) ** 2).groupby(pr.index.normalize()).sum()
    return rv.reindex(dates).to_numpy(float)


def block(sber: pd.DataFrame, sber_ln: np.ndarray, seg: np.ndarray, dates, papers: list,
          imoex: pd.DataFrame | None = None) -> tuple:
    """-> (DataFrame признаков по строкам dates, мета). sber_ln — ln RV_d/w/m SBER (n×3); papers — результаты paper(), включая SBER."""
    dates = pd.DatetimeIndex(dates)
    pxd, pxw, pxm = _ln_dwm(proxy_rv([p["r"] for p in papers], dates), seg)
    A = lambda col: np.column_stack([p["day"][col].reindex(dates).to_numpy(float) for p in papers])  # noqa: E731
    out = pd.DataFrame(index=dates)
    out["proxy_d"], out["proxy_w"], out["proxy_m"] = pxd, pxw, pxm
    for nm, col in (("avg_d", "rv"), ("avg_w", "rv_w"), ("avg_m", "rv_m")):
        out[nm] = pd.DataFrame(np.log(A(col))).mean(axis=1, skipna=True).to_numpy()
    out["gap_med"] = pd.DataFrame(A("gap_rel")).median(axis=1, skipna=True).to_numpy()
    out["jump_share"] = pd.DataFrame(A("jump")).mean(axis=1, skipna=True).to_numpy()
    for k, nm in enumerate(("d", "w", "m")):
        out["rel_" + nm] = sber_ln[:, k] - out["proxy_" + nm].to_numpy()
    meta = {"imoex_from": None}
    if imoex is not None:
        idly, _ = daily.build_daily(imoex, index_days=None)   # IMOEX сам задаёт календарь
        ix = idly["rv"].reindex(dates).to_numpy(float)
        w = idly["rv_w"].reindex(dates).to_numpy(float)
        m = idly["rv_m"].reindex(dates).to_numpy(float)
        meta["imoex_from"] = str(idly.index[0].date())
        for k, (nm, v) in enumerate((("d", ix), ("w", w), ("m", m))):   # день SBER без годного дня IMOEX -> NaN (Ж-201)
            out["imoex_" + nm] = np.log(v)
            out["relx_" + nm] = sber_ln[:, k] - out["imoex_" + nm].to_numpy()
        meta["n_nan_imoex_d"] = int(np.isnan(ix).sum())
    return out, meta


def load_papers(bars_dir: Path, names, sber: pd.DataFrame) -> list:
    return [paper(sber)] + [paper(pd.read_parquet(Path(bars_dir) / (n + ".parquet"))) for n in names]
