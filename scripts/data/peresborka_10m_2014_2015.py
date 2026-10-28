"""Пересборка 10-минутного ряда 2014–2015 из 1-минутных файлов (Ж-116 п. 1), 17 тикеров.

Вход (только чтение, основная копия): data/regen/D25/bars1m/{T}_{2014,2015}.csv.
Календарь баров как в проекте (fastbt.daily_1850): начало бара в [09:50; 18:50); ведро = пол метки минуты до 10 минут.
Агрегация: open — первая минута, high — максимум, low — минимум, close — последняя, volume — сумма.
Выход (своя копия): data/regen/candles_10m_2014_2015/{T}.parquet, sha256.txt, sverka.csv.
Допуск сверки объявлен до счёта: цены |Δ| <= 1e-9, объём — точно. «Полный» день — день, где число баров
в пересобранном и в сверяемом ряду одинаково; на таких днях бар в бар обязано совпасть всё.
Запуск: python3 scripts/data/peresborka_10m_2014_2015.py (без аргументов — пересборка и сверка).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

OPEN_MIN = 9 * 60 + 50   # как fastbt.daily_1850.OPEN_MIN (равенство проверяет тест)
CUT_MIN = 18 * 60 + 50   # как fastbt.daily_1850.CUT_MIN
MAIN = Path(os.environ.get("PERESBORKA_MAIN", "/Users/nilysenok/Desktop/Старое/PhD_Vol_trading"))  # только чтение

TICKERS = sorted(["AFLT", "ALRS", "HYDR", "IRAO", "LKOH", "LSRG", "MGNT", "MOEX",
                  "MTLR", "MTSS", "NVTK", "OGKB", "PHOR", "RTKM", "SBER", "TATN", "VTBR"])
END = pd.Timestamp("2016-01-01")
SVERKA_FROM = pd.Timestamp("2014-08-26")
TOL_PRICE = 1e-9
PRICES = ("open", "high", "low", "close")
OUT = Path("data/regen/candles_10m_2014_2015")
SOURCES = {"raw": ("moex_discovery", "data", "raw", "candles_10m"),
           "final_v6": ("moex_discovery", "data", "final_v6", "candles_10m", "stocks")}


def agg_1m_to_10m(m: pd.DataFrame, cut_min: int = CUT_MIN) -> pd.DataFrame:
    """1-мин бары (begin, open, high, low, close, volume) -> 10-мин (timestamp, OHLCV), окно [OPEN_MIN; cut_min)."""
    b = m.copy()
    b["timestamp"] = pd.to_datetime(b["begin"]).dt.floor("10min")
    mod = b["timestamp"].dt.hour * 60 + b["timestamp"].dt.minute
    b = b[(mod >= OPEN_MIN) & (mod < cut_min) & (b["timestamp"] < END)].sort_values("begin")
    g = b.groupby("timestamp", sort=True)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum().astype("int64")})
    return out.reset_index()


def read_1m(tiker: str, main: Path) -> pd.DataFrame:
    d = main / "data" / "regen" / "D25" / "bars1m"
    return pd.concat([pd.read_csv(d / f"{tiker}_{y}.csv") for y in (2014, 2015)], ignore_index=True)


def read_source(name: str, tiker: str, main: Path) -> pd.DataFrame:
    s = pd.read_parquet(main.joinpath(*SOURCES[name]) / f"{tiker}.parquet")
    mod = s["timestamp"].dt.hour * 60 + s["timestamp"].dt.minute
    return s[(s["timestamp"] < END) & (mod >= OPEN_MIN) & (mod < CUT_MIN)].reset_index(drop=True)


def _per_day(x: pd.DataFrame) -> pd.Series:
    return x.groupby(x["timestamp"].dt.normalize()).size()


def compare(new: pd.DataFrame, src: pd.DataFrame) -> dict:
    """Сверка с допуском: полные дни (равное число баров) и все общие метки, начиная с SVERKA_FROM."""
    a = new[new["timestamp"] >= SVERKA_FROM].set_index("timestamp")
    s = src[src["timestamp"] >= SVERKA_FROM].set_index("timestamp")
    ca, cs = _per_day(a.reset_index()), _per_day(s.reset_index())
    eq_days = ca.index[ca.values == cs.reindex(ca.index).fillna(-1).values]

    def bad(x, y):
        j = x.join(y, rsuffix="_s", how="inner")
        if j.empty:
            return 0
        p = sum((np.abs(j[c] - j[c + "_s"]) > TOL_PRICE) for c in PRICES)
        return int(((p > 0) | (j["volume"] != j["volume_s"])).sum())

    in_eq = lambda ix: ix.normalize().isin(eq_days)
    return {"bars_rebuilt": len(a), "bars_source": len(s), "common": len(a.index.intersection(s.index)),
            "only_rebuilt": len(a.index.difference(s.index)), "only_source": len(s.index.difference(a.index)),
            "days_equal": len(eq_days), "days_rebuilt": len(ca),
            "bad_on_equal_days": bad(a[in_eq(a.index)], s[in_eq(s.index)]), "bad_common": bad(a, s),
            "ge30_rebuilt": int((ca >= 30).sum()), "ge30_source": int((cs >= 30).sum())}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    main_root = MAIN
    OUT.mkdir(parents=True, exist_ok=True)
    rows, shas = [], []
    for t in TICKERS:
        new = agg_1m_to_10m(read_1m(t, main_root))
        p = OUT / f"{t}.parquet"
        new.to_parquet(p, index=False)
        shas.append(f"{sha256(p)}  {p.name}  rows={len(new)}")
        for name in SOURCES:
            rows.append({"ticker": t, "source": name, **compare(new, read_source(name, t, main_root))})
    (OUT / "sha256.txt").write_text("\n".join(shas) + "\n", encoding="utf-8")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sverka.csv", index=False)
    print(df.groupby("source")[["bars_rebuilt", "bars_source", "only_rebuilt", "only_source",
                                "bad_on_equal_days", "bad_common"]].sum().to_string())


if __name__ == "__main__":
    main()
