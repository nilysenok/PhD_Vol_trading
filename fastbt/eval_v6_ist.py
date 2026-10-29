"""Источники рядов EvaluatorV6 (вынесено из eval_v6.py, V6-12d: файл ≤ 200 строк). Данных не читает."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass
class IstochnikiV6:
    """Готовые ряды (≤ 2021). dannye(t, td) → kontekst_v6.Dannye на td.dates; rel(t, rezhim) → дневной td_rel (колонки
    daily_1850.COLS, rel_ryad.td_rel_rezhim); vn(t, faktor) → (z, beta) pd.Series по датам; cbr/div — таблицы sobytiya_e1."""
    dannye: Callable | None = None
    rel: Callable | None = None
    vn: Callable | None = None
    cbr: pd.DataFrame | None = None
    div: pd.DataFrame | None = None
    div_verified: bool = False
    sha: str = ""                  # Д5: хэш всех рядов источников; входит в ключ точки (istochniki_v6.sha_ryadov)
    rel_progrev: bool = True     # дни сетки до первого дня td_rel (нет 10-минуток) — NaN, а не отказ (istochniki_v6 ставит True)
