"""EX0 (R4, G3): ориентиры из сохранённых артефактов — пути, sha256, пересчитанное значение, допуск 5e-5 (корреляция 5e-3). Без обучения, не trials."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s5 import common
from models_v2.s5.c import mix
from models_v2.s6 import paths6
from models_v2.s6.e import comp, crit_e, pe
from models_v2.s6.xgb import c_calc, c_space

TOL, TOL_CORR = 5e-5, 5e-3


def sources() -> dict:
    """{id: [файлы-источники]} — по коду `comp`/`common`."""
    x2_tid, _ = comp.x2_moves()
    xf_tid = c_space.trial_id("l2", {})
    return {"HAR": [common.OUT / "forecasts_p1.parquet"], "XJ": [c_calc.DATA / ("raw_%s_s42.parquet" % x2_tid)],
            "XF": [c_calc.DATA / ("raw_%s_s42.parquet" % xf_tid)], "LF": [comp.lgbm_path("LF", 42)], "B2X": [common.OUT / "fix_forecasts_c.parquet"],
            "B2L": [comp.lgbm_path("B2L", 42)]}


def stage(_a=None) -> int:
    df = comp.load()
    ctx, rv = comp.ctx_of(df), df["rv"].to_numpy(float)
    q = {k: float(crit_e.qlike_series(rv, comp.col(df, k)).mean()) for k in ("HAR", "XJ", "XF", "LF", "B2X", "B2L")}
    h3 = float(crit_e.qlike_series(rv, mix.h3("roll", ctx, comp.col(df, "XF"))).mean())
    ea, eb = np.log(rv / comp.col(df, "XF")), np.log(rv / comp.col(df, "LF"))
    ce = float(np.corrcoef(ea, eb)[0, 1])
    cq = float(np.corrcoef(crit_e.qlike_series(rv, comp.col(df, "XF")), crit_e.qlike_series(rv, comp.col(df, "LF")))[0, 1])
    src = sources()
    spec = [("HAR", "HAR-J(б)", 0.2337, q["HAR"], TOL), ("XJ", "x2_best_j (XGB final)", 0.211295, q["XJ"], TOL), ("XF", "ctrl_f4 (XGB F4)", 0.212538, q["XF"], TOL),
            ("LF", "C0_inh_F4 (LGBM F4)", 0.2117, q["LF"], 5e-4), ("B2X", "Б-2 (baseline XGB)", 0.2205, q["B2X"], 5e-4), ("B2L", "LGBM-аналог Б-2", 0.2223, q["B2L"], 5e-4),
            ("c2_h3_roll_f4", "HAR+XGB F4, roll252 (c2_h3_roll_f4)", 0.2098, h3, TOL), ("corr_qlike_XF_LF", "корреляция дневных потерь QLIKE XGB F4 / LGBM F4 (как в отчётах Ж-206 C/D)", 0.998, cq, TOL_CORR),
            ("corr_lnerr_XF_LF", "корреляция ln-ошибок XGB F4 / LGBM F4 (ориентир «0,998» относился к потерям, не к ln-ошибкам; сравнение — справочное)", 0.998, ce, 1.0)]
    rows = []
    for k, nm, exp, got, tol in spec:
        fl = src.get(k, [])
        rows.append({"id": k, "name": nm, "expected": exp, "recomputed": got, "abs_diff": abs(got - exp), "tol": tol, "ok": bool(abs(got - exp) < tol),
                     "paths": ";".join(paths6.rel(f) for f in fl), "sha256": ";".join(paths6.sha_file(f)[:16] for f in fl), "n": len(df)})
    out = pd.DataFrame(rows)
    pe.write(out, pe.ENS / "EX0_REFERENCE.csv", "EX0: ориентиры, пересчёт из сохранённых рядов; расхождения — по колонке ok")
    ok = bool(out["ok"].all()) and len(df) == paths6.N_GRID and bool((df[["HAR", "XJ", "XF", "LF"]] > 0).all().all())
    print("G3 EX0:", "OK" if ok else "РАСХОЖДЕНИЯ", out[["id", "expected", "recomputed", "ok"]].to_string(index=False), flush=True)
    return 0 if ok else 1
