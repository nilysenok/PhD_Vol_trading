"""Дневной IMOEX для реверса раунда 1 на test1 (общий для листов R-е и R-в, Д7: одно определение).
До STYK (2022-01-01) — дневка из 10-минуток зоны Z2: close = close ПОСЛЕДНЕГО бара дня, open = open первого бара; день, где
последний бар раньше 18:40 (обрыв ленты), не берётся (подставлять нечем, Д4). С STYK — дневные данные index_d (без дыр).
Чтение с фильтром < min(e, HARD) на чтении; строки ≥ 2025 не материализуются."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fastbt import zona  # noqa: E402

POLNYJ_DEN_MIN = 18 * 60 + 40
STYK = pd.Timestamp("2022-01-01")


def dnevka_iz_10min(bars: pd.DataFrame, s: pd.Timestamp, e: pd.Timestamp, kol: str = "close") -> pd.Series:
    """Дневной close (kol='close') или open (kol='open') IMOEX [s, e) из 10-минуток (begin, open, close)."""
    zona.check(e - pd.Timedelta(days=1), "IMOEX 10-мин")
    t = pd.to_datetime(bars["begin"])
    if t.max() >= zona.HARD:
        raise PermissionError(f"IMOEX 10-мин: даты ≥ {zona.HARD.date()}")
    b = pd.DataFrame({"t": t, "v": bars[kol].to_numpy()}).sort_values("t")
    b = b[(b.t >= s) & (b.t < e)].reset_index(drop=True)
    g = b.groupby(b.t.dt.normalize())
    full = g["t"].transform("max").dt.hour * 60 + g["t"].transform("max").dt.minute >= POLNYJ_DEN_MIN
    pick = (g.head(1) if kol == "open" else g.tail(1))
    pick = pick[full.loc[pick.index]]
    return pd.Series(pick["v"].to_numpy(float), index=pd.DatetimeIndex(pick.t.dt.normalize(), name="TRADEDATE"))


def dnevnye_index_d(path: Path, s: pd.Timestamp, e: pd.Timestamp, kol: str = "close") -> pd.Series:
    """Дневной close/open IMOEX [s, e) из index_d (date, open, close); фильтр date < min(e, HARD) — на чтении."""
    import pyarrow.parquet as pq
    zona.check(e - pd.Timedelta(days=1), "IMOEX index_d")
    cut = min(e, zona.HARD).date()
    d = pq.read_table(path, columns=["date", kol], filters=[("date", "<", cut)]).to_pandas()
    d["date"] = pd.to_datetime(d["date"])
    if len(d) and d["date"].max() >= zona.HARD:
        raise PermissionError(f"IMOEX index_d: даты ≥ {zona.HARD.date()}")
    d = d[(d.date >= s) & (d.date < e)].drop_duplicates("date", keep="last")
    return pd.Series(d[kol].to_numpy(float), index=pd.DatetimeIndex(d["date"], name="TRADEDATE")).sort_index()


def zagruzit(path: Path, s: pd.Timestamp, e: pd.Timestamp, path_d: Path | None = None, kol: str = "close") -> pd.Series:
    """[s, e): parquet 10-минуток (begin, open, close); с path_d: [s, STYK) — 10-минутки, [STYK, e) — index_d."""
    import pyarrow.parquet as pq
    if path_d is not None:
        return pd.concat([zagruzit(path, s, STYK, None, kol), dnevnye_index_d(path_d, STYK, e, kol)])
    cut = min(e, zona.HARD)
    bars = pq.read_table(path, columns=["begin", "open", "close"], filters=[("begin", "<", str(cut.date()))]).to_pandas()
    return dnevka_iz_10min(bars, s, e, kol)


def dnevnoj_ryad(path: Path, s: pd.Timestamp, e: pd.Timestamp, path_d: Path | None = None) -> pd.DataFrame:
    """Дневной ряд OPEN/CLOSE (формат ISS-csv, как читает golaya_s7_s9)."""
    return pd.DataFrame({"OPEN": zagruzit(path, s, e, path_d, "open"), "CLOSE": zagruzit(path, s, e, path_d, "close")})
