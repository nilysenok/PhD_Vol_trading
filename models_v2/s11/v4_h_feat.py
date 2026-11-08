"""V4-H: PIT-признаки новых источников информации (ZH_<T>.parquet). Все значения строки d известны на конец дня d; прогноз — на t⁺ = d+1. Поперечные величины — по набору `tickers` (17; в LOTO — 16).
H1: sur_mkt (медиана по бумагам ln(RV_d/H_{d−1}), из Z), sur5 (среднее sur_mkt за 5 дат); H2: sur_sec (то же по бумагам сектора без своей, нет соседей — рыночное); H3: herf, lasth (из Z), firsth (rv_open/rv, M10);
H4: jp_ewma (EWMA, полураспад 20, индикатора «доля скачка > p90 прошлого окна бумаги»), jshare (из Z); H5: ell_dev (|ℓ| − EWMA₂₀|ℓ|), flip5 (смена знака ℓ за 5 строк); H6: brk (CUSUM, `v4_cld_brk`), vol20 (с. о. Δ ln медианной RV за 20 дат).
Прошлое окно процентиля — 252 строки бумаги строго до d, не меньше 60. `lead` — только для теста-мутации («признак дня d+1»); боевое 0."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s10 import hp_paths as HP
from models_v2.s11 import v4_cld_brk as BK, v4_io as IO, v4_z_feat as ZF

SRC = {"H1": ["sur_mkt", "sur5"], "H2": ["sur_sec"], "H3": ["herf", "lasth", "firsth"], "H4": ["jp_ewma", "jshare"], "H5": ["ell_dev", "flip5"], "H6": ["brk", "vol20"]}
ALL = [f for v in SRC.values() for f in v]
NEW = ["sur5", "sur_sec", "firsth", "jp_ewma", "ell_dev", "flip5", "brk", "vol20"]
HL, WIN, MINW, Q_JUMP = 20, 252, 60, 0.9


def cfg_feats(cfg: str) -> list:
    return ALL if cfg in ("H7", "H8") else SRC[cfg]


def sector_surprise(Zl: pd.DataFrame, tickers) -> pd.Series:
    """Медиана ln(RV_d/H_{d−1}) по бумагам того же сектора из `tickers` без своей; нет соседей — рыночный sur_mkt строки. Индекс — как у Zl."""
    L = Zl.pivot(index="date", columns="ticker", values="lerr_h")
    out = pd.Series(np.nan, index=Zl.index)
    for t, g in Zl.groupby("ticker"):
        peers = [p for p in tickers if p != t and p in L.columns and HP.SECTOR.get(p) == HP.SECTOR.get(t)]
        if peers:
            med = L[peers].median(axis=1)       # NaN только если все соседи NaN
            out.loc[g.index] = med.reindex(g["date"]).to_numpy()
        else:
            out.loc[g.index] = g["sur_mkt"].to_numpy()
    return out


def _own(g: pd.DataFrame) -> pd.DataFrame:
    js = g["jshare"]
    past = js.shift(1).rolling(WIN, min_periods=MINW)
    ind = (js > past.quantile(Q_JUMP)).astype(float).where(js.notna() & past.count().ge(MINW))
    a = g["abs_ell"]
    s = np.sign(g["ell"])
    flip = (s != s.shift(1)).astype(float).where(s.shift(1).notna())
    return pd.DataFrame({"jp_ewma": ind.ewm(halflife=HL).mean(), "ell_dev": a - a.ewm(halflife=HL).mean(), "flip5": flip.rolling(5, min_periods=1).max()}, index=g.index)


def date_level(Zl: pd.DataFrame) -> pd.DataFrame:
    """Признаки уровня даты: sur5, brk, vol20 (по поперечным медианам Zl, уже пересчитанным по нужному набору бумаг)."""
    sur = Zl.groupby("date")["sur_mkt"].first().sort_index()
    x = BK.median_series(Zl)
    return pd.DataFrame({"sur5": sur.rolling(5, min_periods=5).mean(), "brk": BK.cusum(x)[0], "vol20": x.diff().rolling(20, min_periods=20).std()})


def build(Zl: pd.DataFrame, firsth: pd.DataFrame, tickers=None, lead: int = 0) -> pd.DataFrame:
    """Zl — длинная Z (ticker, date, t_plus, ell, abs_ell, lerr_h, jshare, lasth, herf, med_lrv, sur_mkt …), поперечные колонки — по `tickers`; firsth — (ticker, date, firsth).
    -> длинная таблица (ticker, date, t_plus, ALL), строки в порядке Zl, отсортированной по (ticker, date)."""
    tk = sorted(Zl["ticker"].unique()) if tickers is None else list(tickers)
    ZF.assert_cross_set(tk)                       # поперечные величины — только по 17 (E11 не входит, У-25)
    Z =Zl.sort_values(["ticker", "date"]).reset_index(drop=True).merge(firsth, on=["ticker", "date"], how="left")
    Z = pd.concat([Z, pd.concat([_own(g) for _, g in Z.groupby("ticker", sort=True)])], axis=1)
    Z["sur_sec"] = sector_surprise(Z, tk)
    Z = Z.join(date_level(Z), on="date")
    if lead:
        Z[ALL] = Z.groupby("ticker")[ALL].shift(-lead)
    return Z[["ticker", "date", "t_plus"] + ALL]


def firsth_table(tickers=None) -> pd.DataFrame:
    """Доля первого часа основной сессии: rv_open / rv дня (M10, `daily_<T>.parquet`, только чтение); определение частей — Ж-214 (`hv2_sched`, OPEN = begin < 11:00)."""
    rows = []
    for t in tickers or IO.T17:
        d = IO.load_daily(t)
        rows.append(pd.DataFrame({"ticker": t, "date": d.index, "firsth": (d["rv_open"] / d["rv"]).where(d["rv"] > 0).to_numpy()}))
    return pd.concat(rows, ignore_index=True)
