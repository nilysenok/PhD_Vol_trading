"""Д17.2: метки баров карантина 28 тикеров для крейта moex_iss (модуль range).

Карантин — функция пайплайна detect_quarantine_flags (moex_discovery/src/pipeline_d001_d003.py), тем же способом,
что d121_rv28.py: полный ряд ISS bars_10m_full с DATE_FROM = 2014-06-09. Выход: data/iss_check/quarantine_28.parquet
(ticker, ts — begin МСК, timestamp[ms]).
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d121_rv28 as R  # noqa: E402


def one(t):
    full = R.load_full(t)
    q = R.quarantine_ts(full[full.timestamp >= pd.Timestamp(R.Q_FROM)].reset_index(drop=True))
    return pd.DataFrame({"ticker": t, "ts": pd.to_datetime(sorted(q), unit="ms").astype("datetime64[ms]")})


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=3) as ex:
        out = pd.concat(list(ex.map(one, R.T28)), ignore_index=True)
    p = R.ISS / "quarantine_28.parquet"
    out.to_parquet(p, index=False)
    print(p, len(out), out.groupby("ticker").size().to_dict())
