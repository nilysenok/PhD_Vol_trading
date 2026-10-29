"""Время 5-минутных баров (прогон этапа 1, Л-3): бар до часа hhmm и план исполнения выхода по `t_vyhod`.

`Day.t` — начало каждого непустого бара дня, минуты от полуночи (заполняет `etap1_data.build_days`); None — времени нет.
Бар «кончается не позже hhmm», если t + 5 ≤ hhmm. Короткий день (последний бар начинается раньше hhmm − 60, то есть
день не доходит до часа до hhmm) → None: выход идёт по close Н6 дня, как и без оси `t_vyhod`.
"""
from __future__ import annotations

import numpy as np

STEP = 5                                   # длина бара, минуты
HORIZON = 60                               # день должен доходить до hhmm − 60 (15:00 → 14:00)
HHMM = {"1500": 900, "1700": 1020, "open2": 600}   # 15:00, 17:00 и 10:00 (open2) — минуты от полуночи
PRICHINY = ("signal", "vremya")


def bar_at(day, hhmm: int) -> int | None:
    """Ж-086 п. 4: индекс бара, начинающегося ровно в hhmm; нет такого — первый бар, начинающийся позже, в тот же день.

    Выход идёт по open этого бара; стоп и тейк проверяются строго до него. Позже hhmm баров нет → len(t): вызывающий
    переносит выход на open первого бара следующего торгового дня (умолчание дирижёра). None — времени нет (Day.t).
    """
    t = getattr(day, "t", None)
    if t is None or len(t) == 0:
        return None
    return int(np.searchsorted(np.asarray(t), hhmm, side="left"))


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
    open2: по сигналу → (1, 600): open бара 10:00 дня x+1 (Ж-086 п. 4; `bar_at`); по τ → (0, None): close дня истечения
    срока (Ж-137 п. 6: срок известен заранее, решения на закрытии дня достаточно; Д2 не нарушена). Если в день y и
    сигнал, и срок — выигрывает срок: close y раньше open y+1. 1500/1700: по τ — open бара hhmm дня истечения (срок
    известен заранее), по сигналу — x+1.
    """
    if prichina not in PRICHINY:
        raise ValueError(f"причина выхода {prichina!r}: допустимо {PRICHINY}")
    if t_vyhod == "close":
        return 0, None
    if t_vyhod == "open2":
        return (0, None) if prichina == "vremya" else (1, HHMM["open2"])
    if t_vyhod not in HHMM:
        raise ValueError(f"t_vyhod {t_vyhod!r}: допустимо close, 1500, 1700, open2")
    return (0 if prichina == "vremya" else 1), HHMM[t_vyhod]
