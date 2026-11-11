"""S-Z-EVAL: форматирование отчёта `S_Z_REPORT.md` (`z_doc`): таблицы markdown, подписи с границами дат и n. Все числа — из CSV `S_Z_*.csv`; здесь ни одного вписанного числа результата.
Публичное API: `md(df, fmt)`, `sgn(x)`, `ci(d, lo, hi)`, `cap(first, last, n)`, `tm(seconds)`, `pick(df, **kw)` (ровно одна строка), `pv(x)`."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd


def md(df: pd.DataFrame, fmt: dict) -> str:
    cols = list(fmt)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join((f % r[c]) if isinstance(f, str) and pd.notna(r[c]) and not isinstance(r[c], str) else "—" if pd.isna(r[c]) else str(r[c]) for c, f in fmt.items()) + " |")
    return "\n".join(out)


def sgn(x) -> str:
    return "—" if pd.isna(x) else "%+.4f" % x


def ci(d, lo, hi) -> str:
    return "—" if pd.isna(d) else "%+.4f" % d if pd.isna(lo) else "%+.4f [%+.4f; %+.4f]" % (d, lo, hi)


def fv(x) -> str:
    """Число — «%+.4f», иное — как есть."""
    try:
        return "%+.4f" % float(x)
    except (TypeError, ValueError):
        return str(x)


def pv(x) -> str:
    return "—" if pd.isna(x) else "%.4f" % x


def cap(first, last, n) -> str:
    return "t⁺ %s…%s, n = %d" % (first, last, int(n))


def tm(s: float) -> str:
    return dt.datetime.fromtimestamp(s).strftime("%d.%m.%Y %H:%M:%S")


def pick(df: pd.DataFrame, **kw) -> pd.Series:
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    if m.sum() != 1:
        raise ValueError("pick %s: строк %d ≠ 1" % (kw, int(m.sum())))
    return df[m].iloc[0]
