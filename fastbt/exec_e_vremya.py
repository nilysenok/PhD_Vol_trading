"""Время 5-минутных баров (прогон этапа 1, Л-3): бар до часа hhmm и план исполнения выхода по `t_vyhod`.

`Day.t` — начало каждого непустого бара дня, минуты от полуночи (заполняет `etap1_data.build_days`); None — времени нет.
Бар «кончается не позже hhmm», если t + 5 ≤ hhmm. Короткий день (последний бар начинается раньше hhmm − 60, то есть
день не доходит до часа до hhmm) → None: выход идёт по close Н6 дня, как и без оси `t_vyhod`.
"""
from __future__ import annotations

import numpy as np

STEP = 5                                   # длина бара, минуты
HORIZON = 60                               # день должен доходить до hhmm − 60 (15:00 → 14:00)
HHMM = {"1500": 900, "1700": 1020}         # 15:00 и 17:00 — минуты от полуночи
PRICHINY = ("signal", "vremya")


def bar_do(day, hhmm: int) -> int | None:
    """Индекс последнего бара дня, кончающегося не позже hhmm; None — времени нет или день короче hhmm − 60."""
    t = getattr(day, "t", None)
    if t is None or len(t) == 0:
        return None
    t = np.asarray(t)
    if t[-1] < hhmm - HORIZON:
        return None
    ok = np.flatnonzero(t + STEP <= hhmm)
    return int(ok[-1]) if len(ok) else None


def vyhod_plan(t_vyhod: str, prichina: str) -> tuple[int, int | None]:
    """(сдвиг дня, hhmm | None) для выхода по `prichina` (signal | vremya).

    close → (0, None): поведение не меняется. 1500/1700: по τ — день x (известен заранее), по сигналу — x+1 (Д2).
    open2 → (1, None): open первого бара дня x+1; hhmm не нужен, вызывающий берёт open.
    """
    if prichina not in PRICHINY:
        raise ValueError(f"причина выхода {prichina!r}: допустимо {PRICHINY}")
    if t_vyhod == "close":
        return 0, None
    if t_vyhod == "open2":
        return 1, None
    if t_vyhod not in HHMM:
        raise ValueError(f"t_vyhod {t_vyhod!r}: допустимо close, 1500, 1700, open2")
    return (0 if prichina == "vremya" else 1), HHMM[t_vyhod]
