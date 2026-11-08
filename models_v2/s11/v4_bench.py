"""V4-0, G3: ориентир SBER 0,2081 (Ж-211, HAR + XGBoost A1_roll_126) из сохранённого ряда `series_a.parquet` и сопоставление с K1, K2, K3, LGBM-D2 панели на тех же датах."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s11 import v4_io as IO

TARGET, TOL = 0.2081, 1e-4          # TABLE17_SBER_H1.csv: 0.20814004386759064
SERIES = "A1_roll_126"


def _q(rv, f) -> float:
    return float(np.mean(metrics.qlike(np.asarray(rv, float), np.asarray(f, float))))


def load_ref() -> pd.DataFrame:
    """Сохранённые ряды Ж-206/210/211: d, target_date, rv, HAR, XJ, LF (components) и A1_roll_126 (series_a)."""
    from models_v2.s6.e import comp, ser
    c = comp.load()[["d", "target_date", "rv", "HAR", "XJ", "LF"]].copy()
    a = pd.read_parquet(ser.file("a"))
    if len(a) != len(c):
        raise ValueError("series_a и components разной длины")
    c[SERIES] = a[SERIES].to_numpy(float)
    return c


def bench_rows(ref: pd.DataFrame, panel: pd.DataFrame) -> tuple:
    """-> (строки SBER_BENCH, словарь итогов). panel — `v4_io.aligned('SBER')`. Сравнение — на датах ряда Ж-211 ∩ общие даты панели."""
    ref = ref.rename(columns={"d": "date"}).copy()
    ref["date"] = pd.to_datetime(ref["date"])
    ref["target_date"] = pd.to_datetime(ref["target_date"])
    m = ref.merge(panel, on="date", how="inner", suffixes=("_ref", ""))
    rv_bad = int((~np.isclose(m["rv_ref"], m["rv"], rtol=0, atol=0)).sum())
    t_bad = int((m["target_date"] != m["t_plus"]).sum())
    rows = []

    def add(name, src, rv, f, note):
        rows.append({"series": name, "source": src, "n": len(f), "qlike": _q(rv, f), "note": note})
    add(SERIES + " (Ж-211, на всех датах ряда)", "series_a.parquet", ref["rv"], ref[SERIES], "потолок: c, λ выбраны по crit_wf на всей истории; SBER 0,2081 в TABLE17")
    for name, col, src in (("HAR (Ж-211)", "HAR", "components"), ("XJ=XGB (Ж-206)", "XJ", "components"), ("LF=LGBM (Ж-206)", "LF", "components")):
        add(name, src, ref["rv"], ref[col], "все даты ряда Ж-211")
    add(SERIES + " (общие даты)", "series_a.parquet", m["rv_ref"], m[SERIES], "даты ряда Ж-211 ∩ панель")
    for name, col in (("K1 HAR-J (панель)", "h"), ("K2 XGB (панель)", "k2"), ("K3 LGBM (панель)", "k3")):
        add(name, "panel", m["rv"], m[col], "те же даты")
    d = m[np.isfinite(m["d2"])]
    add("LGBM-D2 (панель)", "hyb_lgbm_p17", d["rv"], d["d2"], "те же даты, где D2 определён")
    out = {"repro_diff": abs(rows[0]["qlike"] - TARGET), "n_ref": len(ref), "n_common": len(m), "rv_mismatch": rv_bad, "t_plus_mismatch": t_bad,
           "xj_vs_k2_maxdiff": float(np.abs(m["XJ"] - m["k2"]).max()), "lf_vs_k3_maxdiff": float(np.abs(m["LF"] - m["k3"]).max()), "har_vs_k1_maxdiff": float(np.abs(m["HAR"] - m["h"]).max())}
    out["repro_ok"] = bool(out["repro_diff"] <= TOL)
    out["rv_def_same"] = bool(rv_bad == 0 and t_bad == 0 and out["n_common"] == out["n_ref"])
    for r in rows:
        r.update({k: out[k] for k in ("rv_mismatch", "t_plus_mismatch")})
    return pd.DataFrame(rows), out


def run() -> tuple:
    return bench_rows(load_ref(), IO.aligned("SBER"))
