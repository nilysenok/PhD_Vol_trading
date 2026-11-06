"""Стадии run_a.py (кроме lgb_b2): inventory, origin, baseline, diag, lgb_extra, check, audit, check-readme."""
from __future__ import annotations

import sys

import pandas as pd

from models_v2.s6 import paths6

CONTROL = {"harj_b": 0.2337, "fx_b2_g132_b": 0.2205, "b2_f4": 0.2125, "c1_opt": 0.2136, "c2_h3_roll_f4": 0.2098, "c3_f3calm_rv": 0.2101, "e3_roll": 0.2095}
FULL_ID = ("harj_b", "fx_b2_g132_b", "b2_f4", "c1_opt", "c2_h3_roll_f4", "c3_f3calm_rv", "c3_f3calm_g4", "e3_roll")


def inventory(a) -> int:
    from models_v2.s6 import inv
    p = paths6.REP / "INVENTORY.csv"
    if a.check_sha:
        return check_sha(p)
    if a.check:
        return check_numbers(paths6.read_csv(p))
    df = inv.build(paths6.world())
    inv.write(df)
    print("INVENTORY: %d строк; не подтверждено: %d" % (len(df), int(df["verified"].str.startswith("не подтверждено").sum())))
    return 0


def check_numbers(df: pd.DataFrame) -> int:
    """A2: пересчёт и допуски (R4)."""
    bad = []
    t = df.dropna(subset=["qlike_diff"])
    bad += ["табл. %s: %.2e" % (r.id, r.qlike_diff) for r in t.itertuples() if r.qlike_diff >= 1e-12]
    for k, v in CONTROL.items():
        q = float(df.loc[df.id == k, "qlike_recomputed"].iloc[0])
        if abs(q - v) >= 5e-5:
            bad.append("контроль %s: %.6f vs %.4f" % (k, q, v))
    print("A2: строк с пересчётом %d, без пересчёта %d; нарушений %d %s" % (df["qlike_recomputed"].notna().sum(), df["qlike_recomputed"].isna().sum(), len(bad), bad[:10]))
    return 1 if bad else 0


def check_sha(p) -> int:
    """A3: sha в таблице = sha файла; прежние файлы не изменены (mtime не новее GATES_A.md)."""
    df = paths6.read_csv(p)
    bad = []
    for r in df.itertuples():
        for col_p, col_s in (("forecast_path", "forecast_sha256"), ("code_path", "code_sha256")):
            ps, ss = str(getattr(r, col_p)), str(getattr(r, col_s))
            ps = "" if ps == "nan" else ps
            ss = "" if ss == "nan" else ss
            for path, s in zip([x for x in ps.split("; ") if x], ss.split(";")):
                if paths6.sha_file(paths6.ROOT / path) != s:
                    bad.append((r.id, path))
    fl = [(f, f.stat().st_mtime) for d in (*paths6.OLD_DATA, paths6.OLD_REP) for f in d.rglob("*") if f.is_file()]
    lit = [f for f, t in fl if t > paths6.ref_time()]
    newer = [str(f) for f, t in fl if t > paths6.ref_time(paths6.FIX_COMMIT)]
    print("A3: расхождений sha %d %s; прежних файлов новее коммита GATES_A (буквально): %d; новее коммита правок D %s: %d %s" % (
        len(bad), bad[:5], len(lit), paths6.FIX_COMMIT, len(newer), newer[:5]))
    return 1 if bad or newer else 0


def dispatch(a) -> int:
    st = a.stage
    if st == "inventory":
        return inventory(a)
    mod = {"origin": "origin_stage", "baseline": "baseline", "diag": "diag", "lgb_extra": "lgb_extra", "check": "checks", "audit": "checks",
           "check-readme": "readme", "answers": "answers", "readme": "readme", "xgb_seeds": "xgb_seeds"}[st]
    m = __import__("models_v2.s6." + mod, fromlist=["stage"])
    return m.stage(a)


if __name__ == "__main__":
    sys.exit(0)
