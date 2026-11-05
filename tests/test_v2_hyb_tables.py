"""GB6, G21, G25, G28, G29: таблицы Б и этапов 3–4 — одна сетка 2272, sigma2 > 0, независимый пересчёт QLIKE и DM,
выбор Г0/Г2/Г4/Г5 по правилам GATES, критерий вердикта Б механически, шапки со свежим sha кода."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models_v2 import run_p3, run_p4  # noqa: E402
from models_v2.s3 import leak3, prov3  # noqa: E402

OUT, TAB = run_p4.OUT, run_p4.TAB
pytestmark = pytest.mark.skipif(not (TAB / "p4_all.csv").exists(), reason="нет таблиц этапа 4")


def rd(name):
    return pd.read_csv(TAB / name, comment="#")


def own_q(rv, s2):
    r = np.asarray(rv, float) / np.asarray(s2, float)
    return float(np.mean(r - np.log(r) - 1.0))


@pytest.fixture(scope="module")
def fc():
    return run_p4.load()


def test_one_grid_2272_positive_sigma2_same_dates_and_rv_for_all_rows(fc):
    h = fc[fc.model == "harj_b"]
    assert fc.model.nunique() == 145 and (fc.groupby("model").size() == 2271).all()
    for n, f in fc.groupby("model"):
        assert (f["d"].values == h["d"].values).all() and (f["rv"].values == h["rv"].values).all(), n
    assert (fc["sigma2"] > 0).all() and np.isfinite(fc["sigma2"]).all()
    assert (np.diff(h["d"].values.astype("int64")) > 0).all()


def test_p4_qlike_independent_recompute_all_models_and_slices(fc):
    t = rd("p4_all.csv").set_index(["model", "slice"])
    st = pd.read_parquet(OUT / "stress_p3.parquet").set_index("d")["stress"]
    for n, f in fc.groupby("model"):
        y, tg = f["target_date"].dt.year.to_numpy(), f["target_date"]
        s = f["d"].map(st).to_numpy(bool)
        masks = {"all": y > 0, "y2022": y == 2022, "no2022": y != 2022, "y2023": y == 2023,
                 "mar_jun_2020": ((tg >= "2020-03-01") & (tg <= "2020-06-30")).to_numpy(), "stress_g4": s, "calm_g4": ~s}
        for sl, m in masks.items():
            row = t.loc[(n, sl)]
            assert row["n"] == m.sum() and abs(row["qlike"] - own_q(f["rv"][m], f["sigma2"][m])) < 1e-12, (n, sl)
            assert abs(row["mean_r"] - (f["rv"][m] / f["sigma2"][m]).mean()) < 1e-12
    assert (t.xs("all", level="slice")["n"] == 2271).all()


def test_p4_covers_stress_calm_and_years():
    t = rd("p4_all.csv")
    need = {"all", "no2022", "stress_g4", "calm_g4", "rvi_high", "rvi_low", "mar_jun_2020", "y2022"} | {"y%d" % k for k in range(2017, 2027)}
    assert need <= set(t["slice"]) and t.groupby("model")["slice"].nunique().min() == t["slice"].nunique()


def test_g0_g2_choices_follow_gates_rule_and_tables_record_them():
    p2 = pd.read_parquet(OUT / "forecasts_p2.parquet")
    q = {m: own_q(f["rv"], f["sigma2"]) for m, f in p2.groupby("model") if m.endswith("_b")}
    assert len(q) == 16 and min(q, key=q.get) == run_p3.BEST2
    c = rd("p3_choice.csv")
    g2 = min((m for m in q if "_nohar_" not in m), key=q.get)
    assert c[c.step.str.startswith("Г2")]["model"].iloc[0] == g2
    for step in ("Г4: спокойный эксперт", "Г5: основа"):
        part = c[c.step == step]
        assert part["chosen"].sum() == 1 and part[part.chosen]["qlike_all"].iloc[0] == part["qlike_all"].min()


def test_g1_rows_are_pb_rows_bitwise_and_p3_has_14_models():
    p3, pb_ = pd.read_parquet(OUT / "forecasts_p3.parquet"), pd.read_parquet(OUT / "forecasts_pb.parquet")
    assert p3.model.nunique() == 14
    for g, src in run_p3.G1_SRC.items():
        a = p3[p3.model == g].sort_values("d")["sigma2"].to_numpy()
        b = pb_[pb_.model == src].sort_values("d")["sigma2"].to_numpy()
        assert a.tobytes() == b.tobytes(), g


def test_rvi_not_used_after_2021_and_stress_shares():
    s = rd("p3_stress.csv")
    assert (s[s.year >= 2022]["share_rvi_high"] == 0).all() and (s["share_stress"].between(0, 1)).all()
    assert s["n"].sum() == 2271


@pytest.mark.parametrize("name", ["p3_choice.csv", "p3_w.csv", "p3_stress.csv", "p3_leak.csv", "p4_all.csv", "p4_dm.csv"])
def test_headers_carry_status_inputs_and_current_s3_sha(name):
    head = (TAB / name).read_text(encoding="utf-8").splitlines()[0]
    assert "до аудита" in head and "bars=" in head and "rvi=" in head and "s3 sha256[:16]=" + prov3.code_sha3() in head, name


def leak_cases():
    f = TAB / "p3_leak.csv"
    if not f.exists():
        return []
    t = pd.read_csv(f, comment="#")
    return [(r.model, c) for r in t.itertuples() for c in ("ok_i", "ok_ii", "ok_iii")]


@pytest.mark.parametrize("model,crit", leak_cases())
def test_leak_tolerances_gr_rows(model, crit):
    t = pd.read_csv(TAB / "p3_leak.csv", comment="#").set_index("model")
    r = t.loc[model]
    ok = leak3.flags(model, {v: r[v] for v in leak3.VARS})
    assert bool(r[crit]) == ok[crit]  # флаги в таблице = пересчёт по числам и допускам GATES
    if not ok[crit]:  # красное — красным: допуск не двигается
        pytest.xfail("%s %s красный: oracle/full/perm = %.4f/%.4f/%.4f, work %.4f" % (model, crit, r.oracle, r.full, r.perm, r.work))


def test_pb_all_has_2022_and_no2022_and_verdict_recomputed_mechanically():
    a = rd("pb_all.csv")
    assert {"y2022", "no2022"} <= set(a["slice"])
    q = a.pivot(index="model", columns="slice", values="qlike")
    ref = q.loc["harj_b_p1"]
    p2m = rd("p2_metrics.csv")
    best2 = p2m[p2m.model.str.endswith("_b") & ~p2m.model.str.endswith("_btr")].pivot(index="model", columns="slice", values="qlike").min()
    strong = ["mar_jun_2020", "y2023", "y2024", "y2025"]
    v = rd("pb_verdict.csv").set_index("model")
    sel = [m for m in q.index if "_b__" in m and "__p2_" not in m]
    assert set(v.index) == set(sel)
    for m in sel:
        lim = np.minimum(ref[strong], best2[strong])
        assert bool(v.loc[m, "closes2022"]) == bool(q.loc[m, "y2022"] <= ref["y2022"])
        assert bool(v.loc[m, "strong_no_loss"]) == bool((q.loc[m, strong] <= lim).all())
        assert bool(v.loc[m, "weak_no_loss"]) == bool((q.loc[m, strong] <= ref[strong]).all())


def own_dm(d, lag=5):
    n, u = len(d), d - d.mean()
    lrv = (u * u).sum() / n + 2 * sum((1 - k / (lag + 1)) * (u[k:] * u[:-k]).sum() / n for k in range(1, lag + 1))
    z = d.mean() / math.sqrt(lrv / n)
    return z, math.erfc(abs(z) / math.sqrt(2))


def test_pb_dm_three_best_b_rows_independent_recompute(fc):
    t = rd("pb_dm.csv")
    pbm = rd("pb_all.csv")
    sel = pbm[(pbm.slice == "all") & pbm.model.str.contains("_b__") & ~pbm.model.str.contains("__p2_")].sort_values("qlike")
    assert set(t.model) == set(sel.model.head(3)) and len(t) == 9
    pb_ = pd.read_parquet(OUT / "forecasts_pb.parquet")
    h = fc[fc.model == "harj_b"].reset_index(drop=True)
    for r in t.itertuples():
        f = pb_[pb_.model == r.model].sort_values("d").reset_index(drop=True)
        y = h["target_date"].dt.year
        m = {"all": y > 0, "<=2021": y <= 2021, "2022+": y >= 2022}[r.sample].to_numpy()
        rb, rf = (h["rv"] / h["sigma2"])[m].to_numpy(), (f["rv"] / f["sigma2"])[m].to_numpy()
        z, p = own_dm((rb - np.log(rb)) - (rf - np.log(rf)))
        assert abs(r.stat - z) < 1e-9 and abs(r.p - p) < 1e-9 and r.p_holm >= r.p - 1e-15


def test_p4_dm_independent_recompute_holm_and_set(fc):
    t = rd("p4_dm.csv")
    base = fc[fc.model == "harj_b"].reset_index(drop=True)
    assert set(t["sample"]) == {"all", "<=2021", "2022+"} and t.groupby("sample").size().nunique() == 1
    for r in t.itertuples():
        f = fc[fc.model == r.model].reset_index(drop=True)
        y = base["target_date"].dt.year
        m = {"all": y > 0, "<=2021": y <= 2021, "2022+": y >= 2022}[r.sample].to_numpy()
        rb, rf = (base["rv"] / base["sigma2"])[m].to_numpy(), (f["rv"] / f["sigma2"])[m].to_numpy()
        d = (rb - np.log(rb)) - (rf - np.log(rf))
        z, p = own_dm(d)
        assert abs(r.mean_d - d.mean()) < 1e-12 and abs(r.stat - z) < 1e-9 and abs(r.p - p) < 1e-9 and r.sign == np.sign(r.mean_d)
        assert r.p_holm >= r.p - 1e-15
    top = t[(t["sample"] == "all") & t.why.str.contains("топ-5")]
    assert len(top) == 5 and "harj_b" not in set(t.model)
    for cls in ("boost", "g1", "g2", "g3", "g4", "g5", "B1"):
        assert t.why.str.contains("класс " + cls + r"(?:;|$)").any(), cls
