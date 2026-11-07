"""P2, A4 (У-10): признаки утверждённого XGB (K2 = `x2_best_j`: дизайн Б-2 + F4) на 28 бумагах — ТОЛЬКО построение (мир P1a `hp_k23_world`), без обучения K2.
Каждый столбец дизайна K2 нормируется по прошлому самой бумаги тем же z (`hp_feat.zscore`: расширяющиеся медиана и (q75 − q25)/1,349 по строкам s < t, ≥ 252 наблюдений) -> `z_a4_<j>`.
Дата строки = дата d признака (прогноз на следующий собственный день); строки мира K2, которых нет в панели, отбрасываются при слиянии."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s10 import hp_feat as F, hp_k23_world as KW, hp_paths as P
from models_v2.s6.xgb import cfgx, worldx

PATH = P.OUT / "hyb_a4_feats.parquet"


def ticker_frame(bars: pd.DataFrame, imoex: pd.DataFrame, pool: list, t: str) -> pd.DataFrame:
    """Строки мира бумаги (F.dates) × z-нормированные столбцы дизайна K2."""
    w = KW.build_world(bars, imoex, pool)
    X = np.asarray(worldx.spec_of(w, cfgx.CTRL).X, float)
    idx = pd.DatetimeIndex(w["F"].dates)
    z = {"z_a4_%d" % j: F.zscore(pd.Series(X[:, j], index=idx)).replace([np.inf, -np.inf], np.nan).to_numpy() for j in range(X.shape[1])}   # ±inf (RGSS 26.01.2024, 1 ячейка) — пропуск
    return pd.DataFrame({"ticker": t, "date": idx, **z})


def build(read, tickers: list) -> pd.DataFrame:
    """read(тикер | 'IMOEX') -> бары. Бумаги — по одной (счёт ≈ секунды на бумагу); пул 17 основных строится один раз."""
    imoex, pool = read("IMOEX"), KW.pool_papers(read)
    parts = [ticker_frame(read(t), imoex, pool, t) for t in tickers]
    cols = {tuple(p.columns) for p in parts}
    if len(cols) != 1:
        raise ValueError("A4: число столбцов дизайна K2 различается по бумагам: %s" % sorted(len(c) for c in cols))
    df = pd.concat(parts, ignore_index=True)
    keep = [c for c in df.columns if c.startswith("z_a4_") and df[c].notna().any()]
    return df[["ticker", "date"] + keep]


def build_cached() -> pd.DataFrame:
    """Признаки A4 28 бумаг: из `hyb_a4_feats.parquet` или построение (бары — `hp_k23.read_bars`)."""
    if PATH.exists():
        return pd.read_parquet(PATH)
    from models_v2.s10 import hp_k23
    df = build(hp_k23.read_bars, P.TICKERS)
    df.to_parquet(PATH)
    return df


def cols_of(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith("z_a4_")]
