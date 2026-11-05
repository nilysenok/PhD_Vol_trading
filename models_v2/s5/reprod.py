"""A2: воспроизведение σ̂² Б-2 (s4: форма 'm', признаки с IMOEX, HP года из И2 сетки 132) и HAR-J(б) на 3 датах. Не trial."""
from __future__ import annotations

import json

import numpy as np

from models_v2 import har
from models_v2.s4 import rows, runner
from models_v2.s5 import common, frame

DATES = ("2018-06-01", "2022-06-01", "2025-06-02")
TOL_B2, TOL_HJ = 1e-9, 1e-12


def run() -> dict:
    from models_v2 import run_fix
    F, grid, DZ, _, _ = run_fix.prepare(str(common.IMOEX))
    i2 = run_fix.load_i2("i2_choice_g132")
    dz, c = DZ[frame.DZK], common.controls().set_index("d")
    pan = har.Panel(F.dates, F.seg, F.rv, F.X[:, :4])
    out = {"tol_b2_rel": TOL_B2, "tol_harj_rel": TOL_HJ, "dates": []}
    for t in DATES:
        i = int(grid[np.searchsorted(F.dates[grid], np.datetime64(t))])
        r = runner.block_fit(F, dz, i2[rows.year_of(F, i)], i, [i], False, None, "work")
        s2 = float(np.exp(dz.off[i] + r["f"][0] + r["s2_es"] / 2))
        d = F.dates[i]
        ref_b, ref_h = float(c.loc[d, "s2_b2"]), float(c.loc[d, "s2_harj"])
        hj = har.forecast_at(pan, i)["harj_b"]
        out["dates"].append({"d": str(d)[:10], "s2_b2_refit": s2, "s2_b2_parquet": ref_b, "rel_b2": abs(s2 / ref_b - 1),
                             "s2_harj_refit": hj, "s2_harj_parquet": ref_h, "rel_harj": abs(hj / ref_h - 1), "trees": int(r["k"])})
    out["reproduced"] = all(x["rel_b2"] <= TOL_B2 and x["rel_harj"] <= TOL_HJ for x in out["dates"])
    common.REP.mkdir(parents=True, exist_ok=True)
    (common.REP / "REPRODUCTION_A2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out
