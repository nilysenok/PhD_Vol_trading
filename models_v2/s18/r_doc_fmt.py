"""S-R-EVAL, документ (Ж-254 п. 16; `S_R_SPEC.md`, разд. 6, гейт E.G3): помощники чтения таблиц `S_R_*.csv`. Числа в отчёт попадают ТОЛЬКО из CSV, не из расчёта и не из текста.
Публичное API: `OUT_DIR`, `KEYS` (12 таблиц), `read_all(rep) -> {имя: DataFrame}` (строки `#` пропускаются); `c`, `c_`, `md`, `yn`, `val`, `one`, `stat`, `yrs`, `cnc`, `dm_line`, `word` — из `a_doc_fmt` / `a_doc_s2` без правок;
`confirmed(T)`: число подтверждённых строк: main (Δ < 0, p Холма N < 0,05, p SPA < 0,05) и PROC (колонка confirmed); `skey(T, h, b, item)`: ключ строки CONC; `pv(T, kind, key)`: значение PARAMS."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from models_v2.s18.a_doc_fmt import c, c_, cnc, dm_line, md, one, stat, val, yn, yrs  # noqa: F401
from models_v2.s18.a_doc_s2 import word  # noqa: F401

OUT_DIR = Path(__file__).resolve().parents[2] / "docs/otchety/harj_struct"
KEYS = ("table", "stats", "proc", "abl", "years", "conc", "guard", "spa", "mcs", "power", "params", "sel")


def read_all(rep: Path = OUT_DIR) -> dict:
    return {k: pd.read_csv(rep / ("S_R_%s.csv" % k.upper()), comment="#") for k in KEYS}


def confirmed(T: dict) -> int:
    m = T["stats"][T["stats"]["kind"] == "main"]
    n = int(((m["delta_pct"] < 0) & (m["p_holm_N"] < 0.05) & (m["p_spa_c"] < 0.05)).sum())
    return n + int(T["proc"]["confirmed"].map(lambda v: str(v) == "True").sum())


def skey(T: dict, h: str, b: str, item: str) -> str:
    k = T["conc"]
    return str(k[(k["series"] == h) & (k["base"] == b) & (k["item"] == item)]["key"].iloc[0])


def pv(T: dict, variant: str, kind: str, item: str, key="") -> float:
    p = T["params"]
    r = p[(p["variant"] == variant) & (p["kind"] == kind) & (p["item"] == item) & (p["key"].fillna("") == key)]
    return float(r["value"].iloc[0])
