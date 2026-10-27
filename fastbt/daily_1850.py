"""Дневной ряд OHLCV до 18:50 МСК из 10-минуток (Н6, Ж-029, решение владельца 27.10 17:50, вариант (б)).

Информационное множество проекта едино: всё, что видит стратегия в день d, — до 18:50 дня d.
Старый ряд `ohlcv_daily_full.parquet` группировал 10-минутки по календарной дате без фильтра по
времени; с 22.06.2020 (LSRG — с 03.08.2020) в день попадали вечерние бары до 23:40, и close дня
был ценой 23:40, а не 18:50.

Правило: бар входит в день, если его начало лежит в [09:50; 18:50). Утренняя граница 09:50 —
как в `rv_1850_noquar` (moex_iss/src/rv1850.rs, окно W0950: бар предоткрытия 09:50 входит, как в старых
признаках; вечерние отброшены), поэтому до 19.06.2020 ряд совпадает со старым побайтно.
Последний бар до 18:50 — бар с началом 18:40 (он покрывает аукцион закрытия 18:40–18:50; баров
с началом 18:45 в 10-минутках нет; бар 18:50 встречается единично и отброшен). close = close
последнего бара с началом < 18:50, volume = сумма volume таких баров.
Следствие 09:50: утренние бары (06:50–09:40) с 06.12.2021 в день не входят, в отличие от старого ряда.

Ряд идёт с начала 10-минуток (06.2014: разогрев окна валидации 2017 и индикаторов), по 31.12.2021.
Данные ≥ 2022 не читаются: фильтр по timestamp < END идёт на чтении (pyarrow).
Файл кладётся рядом со старым рядом не в moex_volatility, а в data/fastbt/daily/ (в git не идёт);
старый ряд не перезаписывается. Запуск: `python3 -m fastbt.daily_1850 [--out PATH]`.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

END = pd.Timestamp("2022-01-01")          # исключительно: только 2018–2021
OPEN_MIN = 9 * 60 + 50                    # 09:50 — как rv_1850_noquar
CUT_MIN = 18 * 60 + 50                    # 18:50 — бар с началом ≥ 18:50 в день не входит
COLS = ["ticker", "date", "open", "high", "low", "close", "volume"]


def default_main() -> Path:
    return Path(os.environ.get("FASTBT_MAIN", "/Users/nilysenok/Desktop/Старое/PhD_Vol_trading"))


def aggregate(bars: pd.DataFrame, ticker: str, cut_min: int = CUT_MIN) -> pd.DataFrame:
    """Дневные OHLCV из 10-минуток (колонки timestamp, open, high, low, close, volume)."""
    b = bars.sort_values("timestamp")
    mod = b["timestamp"].dt.hour * 60 + b["timestamp"].dt.minute
    b = b[(mod >= OPEN_MIN) & (mod < cut_min)]
    g = b.groupby(b["timestamp"].dt.normalize(), sort=True)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum().astype("int64")})
    out.index.name = "date"
    out = out.reset_index()
    out.insert(0, "ticker", ticker)
    return out[COLS]


def read_bars(path: Path, end: pd.Timestamp = END) -> pd.DataFrame:
    t = pq.read_table(path, filters=[("timestamp", "<", end.to_pydatetime())])
    return t.to_pandas()


def build(main: Path | None = None, tickers: list[str] | None = None) -> pd.DataFrame:
    main = main or default_main()
    src = main / "moex_discovery" / "data" / "raw" / "candles_10m"
    if tickers is None:
        from fastbt.data import TICKERS
        tickers = TICKERS
    parts = [aggregate(read_bars(src / f"{t}.parquet"), t) for t in tickers]
    d = pd.concat(parts, ignore_index=True)
    d = d[d["date"] < END]
    return d.sort_values(["ticker", "date"]).reset_index(drop=True)


def out_path(main: Path | None = None) -> Path:
    return (main or default_main()) / "data" / "fastbt" / "daily" / "ohlcv_daily_1850.parquet"


def ensure(path: Path, main: Path | None = None) -> Path:
    """Собрать ряд, если файла нет (data/fastbt в git не идёт); существующий не перезаписывается."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        build(main).to_parquet(path, index=False)
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    p = a.out or out_path()
    ensure(p)
    print(p, sha256(p))
