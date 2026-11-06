"""Seed-контроли (R7, G8; не trials): seed 42–46 для компонентов, лучших A1…A4, лучших B1/B2, лучшего C. A, C — правила на seed-членах; B — переобучение (seed 43–46).
Выход: SEEDS_E.csv (QLIKE и crit_wf по seed, среднее, разброс) и seeds_e.parquet (ряды)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s6.e import comp, crit_e, pe, resid, rules, ser, st_a, st_b, st_c

SEEDS = comp.SEEDS
FILE = pe.OUT / "seeds_e.parquet"


def finalists(df) -> dict:
    """{id: (kind, параметры)}: лучшие по crit_wf из A1…A4 (каждая), B1, B2 (без абляций), C."""
    fin = {}
    for c in st_a.COMBOS:
        n = st_c.best_of(df, [x for x in df.columns if x.startswith(c + "_")])
        fin[n] = ("A", c, n[len(c) + 1:])
    for f in "12":
        n = st_c.best_of(df, [x for x in st_c.MAIN_B if x[1] == f])
        fin[n] = ("B", f, n)
    n = st_c.best_of(df, ["C2", "C3", "C4", "C5", "C6"])
    fin[n] = ("C", n, n)
    return fin


def stage(_a=None) -> int:
    cdf, df = comp.load(), ser.load()
    ctx, fin = comp.ctx_of(cdf), finalists(df)
    ser_s = {k: [comp.col(cdf, k, s) for s in SEEDS] for k in ("HAR", "XJ", "XF", "LF", "B2X", "B2L")}
    W = resid.build_world()
    X0, ter, chg = W["dz"].X, *st_c.groups(W)
    ad = pd.read_csv(pe.ENS / "ADAPTIVE_RESULTS.csv", comment="#")
    ml, bb = ad["base_ml"].iloc[0], ad["base_b"].iloc[0]
    bser = {}
    for n, (kind, a1, a2) in fin.items():
        if kind == "A":
            ser_s[n] = [rules.sigma2(a2, comp.stack(cdf, st_a.COMBOS[a1], s), ctx) for s in SEEDS]
    for n, (kind, f, _) in fin.items():
        if kind == "B":
            ser_s[n] = [df[n].to_numpy(float)] + [st_b.run_one(W, n, st_b.KIND[n], X0, seed=s)["sigma2"].to_numpy(float) for s in SEEDS[1:]]
            bser[n] = ser_s[n]
    for n, (kind, _, _) in fin.items():
        if kind == "C":
            ser_s[n] = [st_c.build(ctx, comp.col(cdf, "HAR"), comp.col(cdf, ml, s), bser[bb][i], ter, chg)[0][n] for i, s in enumerate(SEEDS)]
    rows = []
    for n, lst in ser_s.items():
        q = [crit_e.qlike_series(cdf["rv"].to_numpy(float), v) for v in lst]
        qa, cw = [float(x.mean()) for x in q], [crit_e.cw(x, cdf["target_date"]) for x in q]
        rows.append({"id": n, "kind": fin.get(n, ("component",))[0], **{"qlike_s%d" % s: v for s, v in zip(SEEDS, qa)}, "qlike_mean": float(np.mean(qa)), "qlike_sd": float(np.std(qa, ddof=1)),
                     **{"crit_wf_s%d" % s: v for s, v in zip(SEEDS, cw)}, "crit_wf_mean": float(np.mean(cw))})
    out = pd.DataFrame(rows)
    out["status"] = pe.STATUS
    pe.write(out, pe.ENS / "SEEDS_E.csv", "seed 42–46 (контроль, не trials); A, C — правила на seed-членах (HAR общий), B — переобучение")
    pd.concat([cdf[["d", "target_date", "rv"]]] + [pd.DataFrame({"%s_s%d" % (n, s): v for s, v in zip(SEEDS, lst)}) for n, lst in ser_s.items()], axis=1).to_parquet(FILE)
    print("G8 seeds:", out[["id", "qlike_mean", "qlike_sd"]].round(5).to_string(index=False), flush=True)
    return 0
