"""Тесты утечки (i)–(iii) этапа 3 для Г1, Г3–Г5 по допускам GATES («Объявлено до счёта (этап 3)»). Допуски не двигать."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s3 import chain

VARS = ("work", "oracle", "full", "perm")
TOL_FULL = 1e-12
ORACLE_MAX = 0.5


def flags(name: str, q: dict) -> dict:
    """q: средний QLIKE по вариантам. oracle <= 0,5·work (Г5: строго < work); full <= work + 1e-12; perm >= work."""
    strict = name.startswith("g5_")
    return {"ok_i": bool(q["oracle"] < q["work"]) if strict else bool(q["oracle"] <= ORACLE_MAX * q["work"]),
            "ok_ii": bool(q["full"] <= q["work"] + TOL_FULL), "ok_iii": bool(q["perm"] >= q["work"])}


def chain_rows(a, ln, choice) -> pd.DataFrame:
    res = {v: chain.build(a, ln, choice, v)[0] for v in VARS}
    out = []
    for name in res["work"]:
        q = {v: float(metrics.qlike(a["rv"], res[v][name]).mean()) for v in VARS}
        out.append({"model": name, **q, **flags(name, q)})
    return pd.DataFrame(out)


def g1_rows(leak_pb: pd.DataFrame, names: dict) -> pd.DataFrame:
    """Г1 = строки Б-1: (i)–(iii) из pb_leak.csv (monthly); добавлено perm >= work (в GB5 не сверялось)."""
    out = []
    for g, b in names.items():
        r = leak_pb[leak_pb.model == b].iloc[0]
        q = {v: float(r[v]) for v in VARS}
        out.append({"model": g, "source": b, **q, **flags(g, q)})
    return pd.DataFrame(out)
