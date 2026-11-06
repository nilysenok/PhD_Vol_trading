"""Ж-212, п. 2 блока: сверка контрольных значений SBER h=1 и их происхождения (файл, SHA-256, конфигурация, код метрики)."""
from __future__ import annotations

import hashlib
import subprocess

import pandas as pd

from models_v2 import metrics
from models_v2.s6.e import comp
from models_v2.s8 import hs_paths as P

DECLARED = {"HAR": 0.2337, "XJ": 0.2113, "LF": 0.2117, "A1_roll_126": 0.2081}
WHAT = {"HAR": "HAR-J(б), PIT: ежедневная OLS на парах t+1 <= d (ctrl s2_harj)", "XJ": comp.NAMES["XJ"], "LF": comp.NAMES["LF"],
        "A1_roll_126": "A1 HAR+XGB, веса roll 126 по закрытым (Ж-210, series_a.parquet)"}


def sha(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=P.ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return "n/a"


def check() -> pd.DataFrame:
    df = comp.load()
    a = pd.read_parquet(P.SERIES_A)
    if not (a["d"].to_numpy() == df["d"].to_numpy()).all():
        raise SystemExit("стоп: сетка series_a != сетка components")
    rows = []
    for k, decl in DECLARED.items():
        s2 = (a if k.startswith("A1") else df)[k].to_numpy(float)
        q = float(metrics.qlike(df["rv"].to_numpy(float), s2).mean())
        src = P.SERIES_A if k.startswith("A1") else P.COMP
        rows.append({"model": k, "qlike": q, "declared": decl, "match_4dp": round(q, 4) == decl, "n": len(df), "d_first": str(df["d"].iloc[0].date()),
                     "d_last": str(df["d"].iloc[-1].date()), "what": WHAT[k], "file": str(src.relative_to(P.ROOT)), "sha256": sha(src),
                     "qlike_def": "models_v2/metrics.py::qlike (r − ln r − 1, r = RV/σ̂²)", "git_head": head()})
    return pd.DataFrame(rows)


def stage(_a=None) -> int:
    t = check()
    P.write_csv(t, "CONTROL_VALUES.csv", "п. 2 блока: сверка контрольных значений; файлы components/series_a в git не отслеживаются (untracked) — происхождение = SHA-256")
    print(t[["model", "qlike", "declared", "match_4dp", "n"]].to_string(index=False), flush=True)
    return 0 if t["match_4dp"].all() else 1
