"""S-A-EVAL, документ (Ж-254 п. 16; `S_A_SPEC.md`, разд. 8): общие помощники форматирования отчёта и чтения таблиц. Числа в отчёт попадают ТОЛЬКО из CSV (`S_A_*.csv`), не из расчёта и не из текста.
Публичное API:
- `OUT_DIR` = docs/otchety/harj_struct; `read_all(rep) -> {имя: DataFrame}` — 9 таблиц (строки `#` пропускаются), ключи `table, stats, years, conc, guard, spa, mcs, power, params`.
- `c(x, nd, sign)`: число с точкой (NaN → «—»; ≥ 1e7 — `%.3g`), `c_` — то же с десятичной запятой (в отчёте); `md(header, rows)`: markdown-таблица; `yn(v)`: «да» / «нет»; `val(v)`: значение условия (число или текст).
- `one(df, **kw)`: единственная строка по ключам.
- `stat(T, hid, kind="main", window="M")`, `yrs(T, series, base, period)`, `cnc(T, series, base, item, key=None)`: строка / значение таблицы по ключу (отказ ValueError, если строки нет или их несколько).
- `dm_line(T)`: строка «D_M: t⁺ …, n = …» из TABLE — стоит перед каждой таблицей отчёта (границы дат и n)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[2] / "docs/otchety/harj_struct"
KEYS = ("table", "stats", "years", "conc", "guard", "spa", "mcs", "power", "params")


def read_all(rep: Path = OUT_DIR) -> dict:
    return {k: pd.read_csv(rep / ("S_A_%s.csv" % k.upper()), comment="#") for k in KEYS}


def c(x, nd: int = 4, sign: bool = False) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return "—"
    return (("%+." if sign else "%.") + str(nd) + "f") % x if abs(x) < 1e7 else "%.3g" % x


def c_(x, nd: int = 4, sign: bool = False) -> str:
    return c(x, nd, sign).replace(".", ",")


def md(header, rows) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(v) for v in r) + " |" for r in rows])


def yn(v) -> str:
    return "да" if (v is True or str(v) == "True") else "нет"


def val(v) -> str:
    try:
        return c_(float(v), 4, True)
    except (TypeError, ValueError):
        return str(v)


def one(df: pd.DataFrame, **kw) -> pd.Series:
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    if m.sum() != 1:
        raise ValueError("ключ %s: строк %d ≠ 1" % (kw, int(m.sum())))
    return df[m].iloc[0]


def stat(T: dict, hid: str, kind: str = "main", window: str = "M") -> pd.Series:
    return one(T["stats"], hypothesis=hid, kind=kind, window=window)


def yrs(T: dict, series: str, base: str, period: str) -> pd.Series:
    return one(T["years"], series=series, base=base, period=period)


def cnc(T: dict, series: str, base: str, item: str, key=None) -> float:
    kw = dict(series=series, base=base, item=item)
    if key is not None:
        kw["key"] = key
    return float(one(T["conc"], **kw)["value"])


def dm_line(T: dict) -> str:
    t = T["table"]
    return "D_M: t⁺ %s…%s, n = %d." % (t["first_tplus"].iloc[0], t["last_tplus"].iloc[0], int(t["n"].iloc[0]))
