"""Вселенная U_d (К-а2, Ж-069 п. 3): топ-k тикеров состава IMOEX по обороту за прошлый календарный год.

sostav(d): последний date_eff <= d (known_from не используется: консервативно по date_eff).
U(d, k): ранг по Σ value за год d.year-1, фиксирован на весь год d.year (ex-ante). Ничья — по тикеру.
Тикер без оборота в прошлом году (нет файла / нет строк / сумма <= 0) в U не входит.
Данные >= 2022 не читаются (отказ). Оборот: {root}/daily_value/{T}.csv (date,value), только чтение.
"""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("VSEL_DATA", "/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/V40"))
SOSTAV = REPO / "docs" / "issledovaniya" / "2026-10-28_sostav_imoex.csv"
LAST_YEAR = 2021


def _year(d) -> int:
    y = pd.Timestamp(d).year
    if y > LAST_YEAR:
        raise ValueError(f"данные >= 2022 не читаются: {d}")
    return y


SOSTAV_COL = "date_eff"  # дата вступления состава в силу (не known_from)


def rank_year(d) -> int:
    """Год, по обороту которого ранжируем: прошлый календарный."""
    return _year(d) - 1


@lru_cache(maxsize=None)
def _sostav_table(path: str, col: str = "date_eff") -> tuple[tuple[pd.Timestamp, frozenset], ...]:
    t = pd.read_csv(path, usecols=[col, "tickers"])
    rows = [(pd.Timestamp(a), frozenset(str(b).split())) for a, b in zip(t[col], t["tickers"])]
    rows.sort(key=lambda x: x[0])
    return tuple(rows)


def sostav_dates(sostav_path: Path | str = SOSTAV) -> list[pd.Timestamp]:
    return [d for d, _ in _sostav_table(str(sostav_path)) if d.year <= LAST_YEAR]


def sostav(d, sostav_path: Path | str = SOSTAV) -> frozenset:
    """Состав IMOEX на d по date_eff; до первой записи — пустой."""
    _year(d)
    ts = pd.Timestamp(d)
    cur: frozenset = frozenset()
    for de, tk in _sostav_table(str(sostav_path), SOSTAV_COL):
        if de <= ts:
            cur = tk
        else:
            break
    return cur


@lru_cache(maxsize=None)
def oborot_goda(ticker: str, year: int, root: str = str(DATA_ROOT)) -> float:
    """Σ value тикера за календарный год; 0.0, если файла или строк за год нет."""
    if year > LAST_YEAR:
        raise ValueError(f"данные >= 2022 не читаются: {year}")
    p = Path(root) / "daily_value" / f"{ticker}.csv"
    if not p.exists():
        return 0.0
    t = pd.read_csv(p, usecols=["date", "value"])
    t = t[pd.to_datetime(t["date"]).dt.year == year]
    return float(t["value"].sum())


def bez_oborota(d, sostav_path: Path | str = SOSTAV, root: Path | str = DATA_ROOT) -> frozenset:
    """Тикеры состава на d без оборота в прошлом году (в U не входят) — для отчёта."""
    y = rank_year(d)
    return frozenset(t for t in sostav(d, sostav_path) if oborot_goda(t, y, str(root)) <= 0)


def U(d, k: int = 40, sostav_path: Path | str = SOSTAV, root: Path | str = DATA_ROOT) -> frozenset:
    y = rank_year(d)
    rows = [(oborot_goda(t, y, str(root)), t) for t in sostav(d, sostav_path)]
    rows = [r for r in rows if r[0] > 0]
    rows.sort(key=lambda r: (-r[0], r[1]))
    return frozenset(t for _, t in rows[:k])


def digest(sostav_path: Path | str = SOSTAV, root: Path | str = DATA_ROOT) -> str:
    """SHA-256 файла состава и всех файлов оборота (по имени, сортировка)."""
    h = hashlib.sha256()
    for p in [Path(sostav_path), *sorted((Path(root) / "daily_value").glob("*.csv"))]:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()
