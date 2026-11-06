"""Ж-214, п. 10 блока: расписание основной сессии по ФАКТИЧЕСКИМ барам и разбиение дня на OPEN / MIDDLE / CLOSE.

Время бара — минуты от полуночи по `begin`. Расписание дня выводится из баров: последний бар основной сессии (begin <= 18:40)
равен 18:30 (сессия 10:00–18:40, 52 бара; режим до апреля 2015) либо 18:40 (10:00–18:50, 53 бара; позже). Режим дня — по его
собственному последнему бару, изолированный сбой на одном дне (отсутствует последний бар) исправляется по соседям.
OPEN — бары с begin < 11:00 (первый час); CLOSE — бары с begin >= последний бар режима − 50 мин (последний час);
MIDDLE — остальное. Границы заданы временем суток, не номером бара: день с дырой или с поздним стартом не сдвигает части.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

START_MIN, BAR_MIN = 600, 10
OPEN_END = START_MIN + 60
CLOSE_SPAN = 60
LASTS = (1110, 1120)          # 18:30, 18:40
MIN_BARS = 30
OPEN, MIDDLE, CLOSE = 0, 1, 2
FULL_POS_FROM = 47            # мутант «номера баров одинаковы для всех лет»: CLOSE = позиции 47..52 (как при 53 барах)


def minutes(begin: pd.Series) -> np.ndarray:
    return (begin.dt.hour * 60 + begin.dt.minute).to_numpy()


def day_schedule(main: pd.DataFrame) -> pd.DataFrame:
    """Бары основной сессии (begin, …) -> по дню: first_min, last_min, n, regime (последний бар режима), ok."""
    m = minutes(main["begin"])
    df = pd.DataFrame({"day": main["begin"].dt.normalize().to_numpy(), "m": m})
    s = df.groupby("day")["m"].agg(first_min="min", last_min="max", n="size")
    raw = s["last_min"].where((s["n"] >= MIN_BARS) & s["last_min"].isin(LASTS))
    s["regime"] = _regime(raw)
    s["ok"] = (s["first_min"] == START_MIN) & (s["last_min"] == s["regime"]) & (s["n"] == (s["regime"] - START_MIN) // BAR_MIN + 1)
    return s


def _regime(raw: pd.Series) -> pd.Series:
    """Причинно (день d видит только дни <= d): 18:40 — если у дня есть такой бар; 18:30 — если и предыдущий классифицированный день
    закончился на 18:30 (или его нет); иначе (прежний режим 18:40, а последнего бара нет) режим остаётся 18:40, день получает ok = False.
    Дни без классификации (короткие, < 30 баров) наследуют режим предыдущего дня; до первого классифицированного — 18:40."""
    r = raw.dropna()
    if r.empty:
        raise ValueError("расписание: нет ни одного дня с последним баром 18:30/18:40")
    prev = r.shift(1)
    reg = r.where((r == LASTS[1]) | prev.isna() | (prev == LASTS[0]), LASTS[1])
    return reg.reindex(raw.index).ffill().fillna(LASTS[1]).astype(int)


def part_by_clock(mins, regime_last, pos=None) -> np.ndarray:
    mins, last = np.asarray(mins), np.asarray(regime_last)
    return np.where(mins < OPEN_END, OPEN, np.where(mins >= last - (CLOSE_SPAN - BAR_MIN), CLOSE, MIDDLE))


def part_by_index(mins, regime_last, pos) -> np.ndarray:
    """МУТАНТ (только для теста): номера баров внутри дня как при 53 барах для всех лет. Должен ловиться тестом расписания."""
    pos = np.asarray(pos)
    return np.where(pos < 6, OPEN, np.where(pos >= FULL_POS_FROM, CLOSE, MIDDLE))


def regime_periods(s: pd.DataFrame) -> pd.DataFrame:
    """Сводка расписания по периодам смены режима и годам: дни, медиана баров, доля штатных дней."""
    g = s.assign(year=s.index.year).groupby(["year", "regime"])
    return g.agg(days=("n", "size"), first_day=("n", lambda x: str(x.index.min().date())), last_day=("n", lambda x: str(x.index.max().date())),
                 n_med=("n", "median"), n_min=("n", "min"), ok_share=("ok", "mean")).reset_index()
