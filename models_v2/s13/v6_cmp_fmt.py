"""V6 лист Z2: форматирование чисел документа (десятичная запятая; владельцу — проценты) и индексы по таблицам. Числа попадают в документ только отсюда, из таблиц."""
from __future__ import annotations

import numpy as np
import pandas as pd


def n4(x) -> str:
    return "—" if x is None or pd.isna(x) else ("%.4f" % x).replace(".", ",")


def n3(x) -> str:
    return "—" if x is None or pd.isna(x) else ("%.3f" % x).replace(".", ",")


def sg(x, d: int = 2) -> str:
    """Знаковое число: «−1,23», «+0,39»; нуль — «0,00»."""
    if x is None or pd.isna(x):
        return "—"
    s = ("%." + str(d) + "f") % abs(x)
    s = s.replace(".", ",")
    return ("−" if x < 0 and float(s.replace(",", ".")) != 0 else "+" if x > 0 and float(s.replace(",", ".")) != 0 else "") + s


def dci(r) -> str:
    """Δ в % уровня K1 [90 % интервал] (DM p)."""
    return "—" if pd.isna(r["delta_pct"]) else "%s [%s; %s] (p %s)" % (sg(r["delta_pct"]), sg(r["lo90"]), sg(r["hi90"]), n3(r["p_one"]))


def holm3(r) -> str:
    return "%s / %s / %s" % (n3(r["p_holm_N722"]), n3(r["p_holm_N3122"]), n3(r["p_holm_Nprog"]))


def win_name(w: str) -> str:
    return {"all": "вся история", "2017_2021": "2017–2021", "2022_2025": "2022–2025", "2026": "2026 (38 дат, неполный)"}[w]


def ix(df: pd.DataFrame, keys: list) -> dict:
    """Индекс строк таблицы по ключам (первая строка при повторе)."""
    return {tuple(r[k] for k in keys): r for _, r in df.drop_duplicates(keys).iterrows()}


def md_table(head: list, rows: list) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def cell_series(s: str, q, mark: str = "") -> str:
    return "%s %s%s" % (s, n4(q), mark)


def finite(x) -> bool:
    return bool(np.isfinite(x))
