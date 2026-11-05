"""C2: H2-L4 на признаках Б-2+F4 (c = 0,5): цель RV/HAR-J(б) (форма `har_h2_l4m`), потеря QLIKE (L4), σ̂² = exp(lb + c·f). Через b1.jobs."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from models_v2.s4 import rows as s4rows
from models_v2.s5.b1 import designs, evalrows, jobs
from models_v2.s5.b1 import world as b1world

NAME, C = "c2_h2l4_f4", 0.5


def spec(w, feats: pd.DataFrame, f4cols) -> designs.Spec:
    specs = designs.build_specs(w.F, w.DZ, w.lb)
    sp = specs["har_h2_l4m"]
    add = feats[list(f4cols)].to_numpy(float)
    assert len(add) == len(sp.X), "строки F4 не совпали с матрицей Б-2"
    return dataclasses.replace(sp, name=NAME, X=np.column_stack([sp.X, add]), cs=(C,))


def run(procs: int = 4) -> tuple:
    """-> (fc, q_loyo, stats) на сетке; фолды и блоки — b1.jobs (ежедневное переобучение, HP года I2)."""
    from models_v2.s5.b2 import world as b2w
    import json
    w = b1world.load()
    feats = pd.read_parquet(b2w.OUT / "feats_b2.parquet")
    f4 = json.loads((b2w.OUT / "accepted.json").read_text())["F4"]
    sp = spec(w, feats, f4)
    specs = {NAME: sp}
    fold = jobs.run(w.F, specs, w.i2, w.eps, "fold", [(NAME, Y, y) for Y in range(2017, 2027) for y in range(2015, Y)], procs)
    ql = evalrows.q_loyo(pd.DataFrame([{"Y": a[1], "y": a[2], "q": q[0]} for a, q in fold.items()]))
    blk = jobs.run(w.F, specs, w.i2, w.eps, "block", [(NAME, int(d)) for d in w.grid], procs)
    r = [blk[(NAME, int(d))] for d in w.grid]
    f, s2 = np.array([x["f"] for x in r]), np.array([x["s2"] for x in r])
    rows_i = np.asarray(w.grid, int)
    sig = designs.sigma2(sp, rows_i, f, s2, w.eps, C)[0]
    fc = pd.DataFrame({"d": w.F.dates[rows_i], "target_date": w.F.dates[rows_i + 1], "sigma2": sig, "rv": w.F.rv[rows_i + 1]})
    tr, es = float(np.mean([x["tr_loss"] for x in r])), float(np.mean([x["es_loss"] for x in r]))
    return fc, ql, {"loss_train": tr, "loss_es": es, "loss_ratio": es / tr, "trees_mean": float(np.mean([x["k"] for x in r])),
                    "n_train_min": int(min(x["n_train"] for x in r))}
