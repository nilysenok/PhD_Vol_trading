"""Вселенная U_d (К-а2, Ж-069 п. 3): топ-k тикеров состава IMOEX по обороту за прошлый календарный год.

sostav(d): последний date_eff <= d (known_from не используется: консервативно по date_eff).
U(d, k): ранг по Σ value за год d.year-1, фиксирован на весь год d.year (ex-ante). Ничья — по тикеру.
Тикер без оборота в прошлом году (нет файла / нет строк / сумма <= 0) в U не входит.
Данные >= 2022 не читаются (отказ), кроме периода wf при отметке снятого замка (_otkryto); за концом состава/оборота — отказ. Оборот: {root}/daily_value/{T}.csv (date,value), только чтение.
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
SOSTAV_2022 = REPO / "docs" / "issledovaniya" / "2026-10-30_sostav_imoex_2022_2026.csv"   # wf: 2022 – 02.2026 (wf-L4a, #750)
LAST_YEAR = 2021


STALE_DAYS = 120   # состав IMOEX меняется поквартально (макс. интервал в таблице — 105 дней); дальше последней строки + 120 — отказ


def _otkryto(y: int) -> None:
    """Год > 2021: только при отметке снятого замка wf и не позже конца периода (период_v2.proverit_chtenie)."""
    from fastbt import period_v2
    try:
        period_v2.proverit_chtenie("wf", otmetka=period_v2.OTMETKA)
    except period_v2.ZamokVladeltsa as e:
        raise ValueError(f"данные >= 2022 не читаются (нет отметки снятого замка): {y}") from e
    if y > period_v2.PERIODY["wf"].konec.year:
        raise ValueError(f"год {y}: позже конца периода wf")


def _year(d) -> int:
    y = pd.Timestamp(d).year
    if y > LAST_YEAR:
        _otkryto(y)
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


def _vidno_s(date_eff: pd.Timestamp, known_from) -> pd.Timestamp:
    """Строка новой таблицы видна не раньше max(date_eff, known_from) (point-in-time, Д3)."""
    return date_eff if pd.isna(known_from) else max(date_eff, pd.Timestamp(known_from))


@lru_cache(maxsize=None)
def _sostav_wf(old: str, new: str) -> tuple[tuple[pd.Timestamp, frozenset], ...]:
    """Прежняя таблица как есть + строки новой с date_eff позже последней прежней (2021-12-17 дублируется — берётся прежняя)."""
    rows = list(_sostav_table(old, SOSTAV_COL))
    t = pd.read_csv(new, usecols=[SOSTAV_COL, "known_from", "tickers"], parse_dates=[SOSTAV_COL, "known_from"])
    for a, k, b in zip(t[SOSTAV_COL], t["known_from"], t["tickers"]):
        if a > rows[-1][0]:
            rows.append((_vidno_s(a, k), frozenset(str(b).split())))
    rows.sort(key=lambda x: x[0])
    return tuple(rows)


def sostav_dates(sostav_path: Path | str = SOSTAV) -> list[pd.Timestamp]:
    return [d for d, _ in _sostav_table(str(sostav_path)) if d.year <= LAST_YEAR]


def sostav(d, sostav_path: Path | str = SOSTAV) -> frozenset:
    """Состав IMOEX на d по date_eff; до первой записи — пустой."""
    _year(d)
    ts = pd.Timestamp(d)
    cur: frozenset = frozenset()
    last = None
    rows = (_sostav_wf(str(sostav_path), str(SOSTAV_2022)) if ts.year > LAST_YEAR and str(sostav_path) == str(SOSTAV)
            else _sostav_table(str(sostav_path), SOSTAV_COL))
    for de, tk in rows:
        if de <= ts:
            cur, last = tk, de
        else:
            break
    if ts.year > LAST_YEAR and last is not None and ts > last + pd.Timedelta(days=STALE_DAYS):
        raise ValueError(f"состав IMOEX на {ts.date()}: последняя запись {last.date()} (> {STALE_DAYS} дней) — состав 2022+ не подключён (Д4)")
    return cur


@lru_cache(maxsize=None)
def oborot_goda(ticker: str, year: int, root: str = str(DATA_ROOT)) -> float:
    """Σ value тикера за календарный год; 0.0, если файла или строк за год нет."""
    if year > LAST_YEAR:
        _otkryto(year)
    p = Path(root) / "daily_value" / f"{ticker}.csv"
    if not p.exists():
        return 0.0
    t = pd.read_csv(p, usecols=["date", "value"])
    t = t[pd.to_datetime(t["date"]).dt.year == year]
    if year > LAST_YEAR and t.empty:
        raise ValueError(f"оборот {ticker} за {year}: в {p.name} нет строк года (0.0 молча не подставляется, Д4)")
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
    from fastbt import zona
    extra = [SOSTAV_2022] if zona.name() == "wf" and str(sostav_path) == str(SOSTAV) else []   # dev: хэш прежний
    for p in [Path(sostav_path), *extra, *sorted((Path(root) / "daily_value").glob("*.csv"))]:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()
