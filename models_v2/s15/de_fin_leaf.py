"""FIN: сверка сводных таблиц с таблицами листов E1–E4 (G16: «числа равны таблицам листов»). Только чтение. Сравнение — по ключам строк, все числовые столбцы (rtol 1e-9; Холм пар не сравнивается: семейство пар в FIN другое)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s15 import de0_io as IO, de_fin_tabs as T

KEYS = {"METRICS": ["series", "window"], "TESTS": ["window", "base"], "GO": ["cond"], "YEARS": ["year"], "STATE": ["series", "window", "state"], "PAIRS": ["x", "y", "window"]}
LEAF_NAME = {"METRICS": "METRICS", "TESTS": "TESTS", "GO": "GO", "YEARS": "YEARS", "STATE": "STATE_QLIKE", "PAIRS": "PAIRS"}
SKIP = {"p_holm_N", "p_holm_Nwide", "hyp", "config"}


def _diff(a: pd.DataFrame, b: pd.DataFrame, keys: list, tag: str) -> list:
    """a — строки FIN, b — строки листа (подмножество по ключам). -> список расхождений."""
    a, b = a.set_index(keys), b.set_index(keys)
    miss = b.index.difference(a.index)
    out = ["%s: в FIN нет строк листа: %d" % (tag, len(miss))] if len(miss) else []
    both = b.index.intersection(a.index)
    for c in b.columns:
        if c in SKIP or c not in a.columns:
            continue
        x, y = a.loc[both, c], b.loc[both, c]
        bad = (~np.isclose(x.astype(float), y.astype(float), rtol=1e-9, atol=1e-12, equal_nan=True)).sum() if pd.api.types.is_numeric_dtype(y) else int((x.astype(str) != y.astype(str)).sum())
        if bad:
            out.append("%s: столбец %s — расхождений %d" % (tag, c, bad))
    return out


def _leaf(k: str, name: str) -> pd.DataFrame:
    return pd.read_csv(IO.REP / ("%s_%s.csv" % (k, name)), comment="#")


def _fin_part(t: dict, name: str, x: str, k: str) -> pd.DataFrame:
    df = t[name]
    if name in ("TESTS", "GO", "YEARS"):
        return df[df["config"] == x]
    if name == "PAIRS":
        return df[df["x"] == x]
    return df


def mismatches(t: dict) -> tuple:
    """-> (число сверенных чисел, список расхождений). Для каждого листа Ek (ряд Xk): METRICS (все ряды листа), TESTS, GO, YEARS, STATE_QLIKE, PAIRS (по x, y, окну)."""
    bad, n = [], 0
    for x, k in T.LEAF.items():
        for name, leaf in LEAF_NAME.items():
            lt = _leaf(k, leaf)
            if name == "PAIRS":
                lt = lt[lt["x"] == x]
            ft = _fin_part(t, name, x, k)
            bad += _diff(ft, lt, KEYS[name], "%s %s" % (k, name))
            n += int(lt.select_dtypes("number").shape[0] * lt.select_dtypes("number").shape[1])
    return n, bad
