"""README_B1.md: блок между маркерами B1:auto строится из таблиц кодом и проверяется (--stage check-readme)."""
from __future__ import annotations

import pandas as pd

from models_v2.s5 import common
from models_v2.s5.b1 import configs

README = common.REP / "README_B1.md"
A, B = "<!-- B1:auto -->", "<!-- /B1:auto -->"
COLS = ["row_id", "q_int_kind", "q_int", "q_int_b2", "qlike_all", "qlike_2017_2021", "qlike_2022", "qlike_2023_2025", "qlike_no_240222",
        "qlike_no_2022", "d_harj", "d_harj_no_240222", "d_b2", "perspective"]


def block() -> str:
    parts = []
    for dirn, fn in configs.TABLE.items():
        df = common.read_comment_csv(common.REP / fn)
        df = df[df.direction == dirn]
        parts.append("**%s** (%d trials)\n\n| %s |\n|%s|\n%s" % (dirn, len(df), " | ".join(COLS), "---|" * len(COLS), "\n".join(
            "| " + " | ".join(("%.4f" % v if isinstance(v, float) else str(v)) for v in r) + " |" for r in df[COLS].itertuples(index=False))))
    return A + "\n" + "\n\n".join(parts) + "\n" + B


def write_auto() -> int:
    txt = README.read_text(encoding="utf-8") if README.exists() else "# B1\n\n" + A + "\n" + B + "\n"
    i, j = txt.index(A), txt.index(B) + len(B)
    README.write_text(txt[:i] + block() + txt[j:], encoding="utf-8")
    return 0


def check_auto() -> int:
    txt = README.read_text(encoding="utf-8")
    cur = txt[txt.index(A):txt.index(B) + len(B)]
    ok = cur == block()
    print("check-readme:", "OK" if ok else "РАСХОЖДЕНИЕ с таблицами")
    return 0 if ok else 1
