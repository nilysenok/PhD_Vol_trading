"""B2: перепроверка таблиц — QLIKE строк пересчитан из прогнозов fc_*.parquet (<= 1e-12), сетка 2271, trials уникальны, sha кода в шапках текущий, Holm по DM к Б-2."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models_v2 import metrics  # noqa: E402
from models_v2.s3 import dm  # noqa: E402
from models_v2.s5 import common  # noqa: E402
from models_v2.s5.b2 import world  # noqa: E402

R = common.REP
bad = 0
sha = world.code_sha()
for f in ("FEATURE_ABLATIONS.csv", "WINDOW_RESEARCH.csv", "FEATURE_IMPORTANCE.csv"):
    head = open(R / f, encoding="utf-8").readline()
    ok = "sha=" + sha in head
    print(f, "sha в шапке текущий:", ok)
    bad += not ok
b2 = common.controls()
for f in ("FEATURE_ABLATIONS.csv", "WINDOW_RESEARCH.csv"):
    t = pd.read_csv(R / f, comment="#")
    for m, q in zip(t["model"], t["qlike_all"]):
        fc = pd.read_parquet(world.OUT / ("fc_%s.parquet" % m))
        assert len(fc) == 2271 and (fc["d"].to_numpy() == b2["d"].to_numpy()).all() and np.array_equal(fc["rv"], b2["rv"]), m
        d = abs(float(metrics.qlike(fc["rv"], fc["sigma2"]).mean()) - q)
        bad += d > 1e-12
        if d > 1e-12:
            print("РАСХОЖДЕНИЕ", m, d)
print("QLIKE пересчитан для всех строк; расхождений:", bad)
t = pd.read_csv(R / "FEATURE_ABLATIONS.csv", comment="#")
t = t[t.model != "BASE"]
print("Holm(12) по DM p к Б-2:", dict(zip(t["model"], dm.holm(t["dloss_b2_dm_p"].to_numpy()).round(4))))
tr = pd.read_csv(R / "OPTUNA_TRIALS_B2.csv")
print("trials:", len(tr), tr.groupby("направление").size().to_dict(), "уникальны:", tr["trial_id"].is_unique)
sys.exit(1 if bad else 0)
