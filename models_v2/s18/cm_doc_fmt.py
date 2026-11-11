"""S-C-EVAL, документ (Ж-254 п. 16; `S_C_SPEC.md`, разд. 7): помощники чтения таблиц `S_C_*.csv`. Числа в отчёт попадают ТОЛЬКО из CSV, не из расчёта и не из текста.
Публичное API: `OUT_DIR`, `KEYS` (10 таблиц: table, stats, years, conc, guard, spa, mcs, power, params, lnm), `read_all(rep) -> {имя: DataFrame}` (строки `#` пропускаются);
`c`, `c_`, `md`, `yn`, `val`, `one`, `stat`, `yrs`, `cnc`, `dm_line`, `word` — из `a_doc_fmt` / `a_doc_s2` без правок (`c_` — десятичная запятая; `word(d)`: «лучше» / «хуже» / «равно» по знаку Δ);
`confirmed(T)`: число подтверждённых строк семейств main и struct (Δ < 0, p Холма N < 0,05 и p SPA < 0,05); `skey(T, h, b, item)`: ключ строки CONC."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from models_v2.s18.a_doc_fmt import c, c_, cnc, dm_line, md, one, stat, val, yn, yrs  # noqa: F401
from models_v2.s18.a_doc_s2 import word  # noqa: F401

OUT_DIR = Path(__file__).resolve().parents[2] / "docs/otchety/harj_struct"
KEYS = ("table", "stats", "years", "conc", "guard", "spa", "mcs", "power", "params", "lnm")


def read_all(rep: Path = OUT_DIR) -> dict:
    return {k: pd.read_csv(rep / ("S_C_%s.csv" % k.upper()), comment="#") for k in KEYS}


def confirmed(T: dict) -> int:
    m = T["stats"][T["stats"]["kind"].isin(["main", "struct"])]
    return int(((m["delta_pct"] < 0) & (m["p_holm_N"] < 0.05) & (m["p_spa_c"] < 0.05)).sum())


def skey(T: dict, h: str, b: str, item: str) -> str:
    k = T["conc"]
    return str(k[(k["series"] == h) & (k["base"] == b) & (k["item"] == item)]["key"].iloc[0])
