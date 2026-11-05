"""GB3/GB5: сквозной прогон run_pb на синтетике (2 расписания, leak-варианты, full), сборка (а)/(б), критерий вывода, DM."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_v2_hyb_leak import synth  # noqa: E402

from models_v2 import har, metrics, run_pb  # noqa: E402
from models_v2.s2 import features as ft  # noqa: E402
from models_v2.s3 import gen, pb  # noqa: E402

PARAMS = {("xgb", "mse"): (3, 0.5), ("lgbm", "mse"): (3, 0.5)}
NAMES = ("b1_xgb_c100", "b1_xgb_c050", "b5_lgbm_free")


@pytest.fixture(scope="module")
def world():
    bars, _ = synth()
    F = ft.build(bars)
    grid = har.eligible(har.Panel(F.dates, F.seg, F.rv, F.X[:, :4]))[-4:]
    D, off = pb.build_designs(F)
    rs = [r for r in pb.rows() if r.name in NAMES]
    return F, grid, D, off, rs, run_pb.run_fits(F, D, grid, PARAMS, rs)


def test_registry_counts_and_unique_fits():
    rs = pb.rows()
    assert len(rs) == 22 and len({r.name for r in rs}) == 22 and len(pb.fit_keys(rs)) == 16
    assert all(r.lib == "lgbm" for r in rs if r.design in ("b3n", "b3h")) and not any(r.mono for r in rs if r.name.startswith("b1"))


def test_run_fits_keys_and_frames_match_direct_run(world):
    F, grid, D, off, rs, res = world
    assert ("monthly", "b1", "xgb", False, "full") in res and ("daily", "b1", "xgb", False, "oracle") not in res
    fc = run_pb.all_frames(F, off, rs, res, "daily")
    assert len(fc) == len(rs) * 2 * len(grid) and (fc["sigma2"] > 0).all() and np.isfinite(fc["sigma2"]).all()
    d = int(grid[-1])
    r = [x for x in rs if x.name == "b1_xgb_c050"][0]
    g = gen.run_design(D["b1"], pb.spec_of("xgb", False), 3, 0.5, d, [d])
    want_b = np.exp(off["b1"][d] + 0.5 * g["f"][0] + g["s2_es"] / 2)
    want_a = np.exp(off["b1"][d] + 0.5 * g["f"][0])
    got = fc[(fc.d == F.dates[d])].set_index("model")["sigma2"]
    assert got["b1_xgb_c050_b__daily"] == pytest.approx(want_b, rel=1e-12) and got["b1_xgb_c050_a__daily"] == pytest.approx(want_a, rel=1e-12)
    assert want_b > want_a and r.bias


def test_leak_table_dm_and_verdict_run(world):
    F, grid, D, off, rs, res = world
    lk = run_pb.leak_table(F, off, rs, res, 0.4)
    assert list(lk.columns) == ["model", "qlike_rv_m", "work", "oracle", "perm", "full"] and len(lk) == len(rs)
    fc = run_pb.all_frames(F, off, rs, res, "daily")
    h = fc[fc.model == "b1_xgb_c100_b__daily"].assign(model="harj_b")
    t = run_pb.dm_table(fc, h, ["b1_xgb_c050_b__daily"])
    assert set(t["sample"]) == {"all", "<=2021", "2022+"} and (t["p_holm"].fillna(1) >= t["p"].fillna(0)).all()


def test_verdict_criterion_strong_vs_weak_and_extra_slice():
    sl = ["y2022", "mar_jun_2020", "y2023", "y2024", "y2025"]
    mt = pd.DataFrame([{"model": m, "slice": s, "qlike": q} for m, qs in
                       (("good", [0.5, 0.2, 0.2, 0.2, 0.2]), ("weak", [0.5, 0.3, 0.2, 0.2, 0.2]), ("no2022", [0.9, 0.1, 0.1, 0.1, 0.1]))
                       for s, q in zip(sl, qs)])
    ref = pd.Series([0.6, 0.35, 0.3, 0.3, 0.3], index=sl)
    best2 = pd.Series([0.9, 0.25, 0.3, 0.3, 0.3], index=sl)   # лучшая строка этапа 2 на 2020 строже HAR-J
    v = pb.verdict(mt, ref, best2).set_index("model")
    assert v.loc["good", "closes_and_strong"] and not v.loc["weak", "strong_no_loss"] and v.loc["weak", "weak_no_loss"]
    assert v.loc["weak", "closes_and_weak"] and not v.loc["no2022", "closes2022"]
    f = pd.DataFrame({"model": "m", "target_date": pd.to_datetime(["2021-05-05", "2022-05-05", "2023-05-05"]),
                      "sigma2": [1.0, 1.0, 1.0], "rv": [1.0, 100.0, 1.0]})
    e = pb.extra_slices(f)
    assert e["n"].iloc[0] == 2 and e["qlike"].iloc[0] == pytest.approx(0.0) and metrics.qlike(100.0, 1.0) > 1
