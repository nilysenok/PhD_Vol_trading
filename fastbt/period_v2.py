"""Период v2 (Ж-113 п. 1, 6; Ж-114 п. 1–2): train 2015–2019, прогрев с 26.08.2014; путь A без σ-источников.

Одна точка подключения правила качества (`vselennaya_kachestvo`, #497): невалидный тикер-день — нет сигнала,
позиция держится (`hold_invalid`; вызывается из `s9._job` только при PERIOD == "v2").
Ранг с окном < 252 (C/D/E) в пути A не участвует — не проверялся здесь.
Дивиденды 2014 не подключены: таблица дивидендов начинается с 2015 (NOTE_DIV_2014); нулём не подменяется —
ряд 2014 идёт без поправки на отсечки и в метрики не входит (только прогрев).
"""
from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PERIOD = "v2"
YEARS = (2015, 2016, 2017, 2018, 2019)
VAL_START = 2014                                   # год прогрева (как 2015 у train4)
START = pd.Timestamp("2014-08-26")                 # начало прогрева: покрытие 10-минуток
NOTE_DIV_2014 = "дивиденды 2014 не подключены"


@dataclasses.dataclass(frozen=True)
class Period:
    """Период как один объект: старт прогрева, первый год сумм, последний день (Б8, Ж-136 п. 1)."""
    name: str
    progrev: pd.Timestamp
    y0: int
    konec: pd.Timestamp


PERIODY = {"train4": Period("train4", pd.Timestamp("2015-01-01"), 2016, pd.Timestamp("2019-12-31")),
           "v2": Period("v2", START, 2015, pd.Timestamp("2019-12-31")),
           "wf": Period("wf", pd.Timestamp("2014-06-01"), 2015, pd.Timestamp("2026-02-28")),   # walk-forward, Ж-136
           # вся история куба v7 (Ж-155 п. 7): прогрев 26.08.2014–31.12.2016, оценка 01.01.2017–31.12.2025 (Ж-159; янв–фев 2026 — справочно); зона данных — wf
           "polnaya": Period("polnaya", START, 2017, pd.Timestamp("2025-12-31"))}
OTMETKA = Path(__file__).resolve().parents[1] / "docs/resheniya/wf_zamok_snyat.json"            # отметку ставит владелец
ZAKRYTO_RE = re.compile(r"(^|/)test_locked(/|$)|_(test|202[2-9])([_./]|$)")                     # как zamki.json: чтение


class ZamokVladeltsa(PermissionError):
    """Период или путь ≥ 2022 до снятия замка test_locked/ владельцем."""


def gody(period: str) -> tuple:
    p = PERIODY[period]
    return tuple(range(p.y0, p.konec.year + 1))


def kvartaly_spisok(period: str) -> list:
    """Метки кварталов сумм от первого года до квартала последнего дня периода (wf: 2015Q1…2026Q1 — 45)."""
    p = PERIODY[period]
    return [f"{y}Q{q}" for y in gody(period) for q in range(1, 5) if (y, q) <= (p.konec.year, p.konec.quarter)]


def otmetka_godna(otmetka) -> bool:
    """Отметка снятого замка годна: файл непустой, корректный JSON-объект с ключами preregistraciya_v2, sha256 (64 hex),
    data_snyatiya (формат акта `scripts/server/akt_snyatiya_zamka.py`). sha с файлом предрегистрации не сверяется."""
    try:
        d = json.loads(Path(otmetka).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (isinstance(d, dict) and bool(d.get("preregistraciya_v2")) and bool(d.get("data_snyatiya"))
            and isinstance(d.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", d["sha256"]) is not None)


def proverit_chtenie(period: str, put=None, otmetka: Path = OTMETKA) -> None:
    """Отказ с понятным сообщением: период с днями ≥ 2022 или путь ≥ 2022/test_locked/ без годной отметки владельца."""
    zakryto = (period in PERIODY and PERIODY[period].konec.year >= 2022) or (put is not None and ZAKRYTO_RE.search(str(put)))
    if zakryto and not otmetka_godna(otmetka):
        raise ZamokVladeltsa(f"период {period!r}/путь {put!r}: данные ≥ 2022 закрыты (замок test_locked/, Ж-136); "
                             f"читать можно после отметки владельца {Path(otmetka).name} — отказ")


def trim_td(td):
    """TickerData с датами >= START (прогрев v2 начинается 26.08.2014, а не с начала дневного ряда)."""
    k = int(np.searchsorted(td.dates, np.datetime64(START)))
    cut = lambda a: a[k:]                          # noqa: E731
    return dataclasses.replace(td, dates=cut(td.dates), open=cut(td.open), high=cut(td.high), low=cut(td.low),
                               close=cut(td.close), volume=cut(td.volume), log_ret=cut(td.log_ret),
                               sigma={h: cut(a) for h, a in td.sigma.items()})


def valid_vec(ticker: str, dates, kach=None, period=None, main=None) -> np.ndarray:
    """Валидность тикер-дня по правилу качества; даты вне календаря 10-минуток — невалидны.
    period="wf": календарь строится до конца wf (при отметке замка); дата позже последнего дня календаря + 3 суток — ValueError,
    а не молчаливое «невалидно» (fill_value=False раньше делал невалидным каждый день ≥ 2020)."""
    if kach is None:
        from fastbt import vselennaya_kachestvo as vk
        kach = vk.default(main, period)
    d = pd.DatetimeIndex(dates)
    if period == "wf" and len(d) and d.max() > kach.dates[-1] + pd.Timedelta(days=3):
        raise ValueError(f"валидность на {d.max().date()}: календарь баров кончается {kach.dates[-1].date()} (молча «невалидно» нельзя)")
    return kach.valid[ticker].reindex(d, fill_value=False).to_numpy(bool)


def hold_invalid(pos: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """В невалидный день позиция = позиции предыдущего дня (нет сигнала — держится). Первый день — как есть."""
    idx = np.where(valid, np.arange(len(pos)), 0)
    idx[0] = 0
    return pos[np.maximum.accumulate(idx)]


def hold_years(a: dict, ticker: str, dates, kach=None, period=None) -> dict:
    """{год: позиции на весь ряд} → то же с удержанием в невалидные дни."""
    ok = valid_vec(ticker, dates, kach, period)
    return {Y: hold_invalid(v, ok) for Y, v in a.items()}


if __name__ == "__main__":      # python3 period_v2.py <отметка>: выход 0 — годна (для perenos/obshchee.sh и manifest_perenos.py)
    import sys
    sys.exit(0 if len(sys.argv) == 2 and otmetka_godna(sys.argv[1]) else 1)
