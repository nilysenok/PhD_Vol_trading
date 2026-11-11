"""Форматирование отчёта S-D-EVAL: таблицы markdown и строка границ дат и n (`d_doc`)."""
from __future__ import annotations

import pandas as pd


def md(head: list, rows: list) -> list:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def bounds(df: pd.DataFrame) -> str:
    return "D_M: t⁺ %s…%s, n = %d" % (df["first_tplus"].iloc[0], df["last_tplus"].iloc[0], int(df["n"].iloc[0]))
