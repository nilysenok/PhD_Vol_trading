"""Ж-198, Б-1…Б-7: доработка бустинга одним прогоном (ежедневное и ежемесячное переобучение).

Запуск: python3 models_v2/run_pb.py   (процессов <= 6). Требует таблиц и прогнозов этапа 2.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models_v2 import har, metrics, prov  # noqa: E402
from models_v2.s2 import boost  # noqa: E402
from models_v2.s2 import features as ft  # noqa: E402
from models_v2.s3 import dm, pb, pool3, post, prov3  # noqa: E402

OUT = prov.ROOT / "data/regen/v2_sber_h1"
TAB = prov.ROOT / "docs/tables/v2_sber_h1"
BEST2 = "xgb_mse_nohar_mono_b"
SCHED = (("daily", True), ("monthly", False))


def params_from_p2() -> dict:
    t = pd.read_csv(TAB / "p2_tuning.csv", comment="#")
    t = t[t["chosen"]]
    return {(r.lib, r.loss): (int(r.depth), float(r.delta)) for r in t.itertuples()}


def run_fits(F, D, grid, params, rs) -> dict:
    """{(расписание, design, lib, mono, variant): [(d_idx, результат)]}; leak-варианты и full — только monthly (как этап 2)."""
    jobs, dmap = [], {}
    for sch, daily in SCHED:
        bl = boost.blocks(F, grid, daily)
        for dn, lib, mono in pb.fit_keys(rs):
            dp, dl = params[(lib, "mse")]
            for v in ("work",) + (("oracle", "perm") if sch == "monthly" else ()):
                for T, ds in bl.items():
                    jobs.append(((sch, dn, lib, mono, v, T), dn, pb.spec_of(lib, mono), dp, dl, T, ds, v))
                    dmap[jobs[-1][0]] = ds
            if sch == "monthly":
                g = [int(x) for x in grid]
                jobs.append(((sch, dn, lib, mono, "full", -1), dn, pb.spec_of(lib, mono), dp, dl, len(F.dates) - 1, g, "work"))
                dmap[jobs[-1][0]] = g
    t0 = time.time()
    res = pool3.map_jobs(D, jobs)
    print("счёт Б: %d задач, %.1f мин" % (len(jobs), (time.time() - t0) / 60))
    out = {}
    for key, r in res:
        out.setdefault(key[:5], []).append((dmap[key], r))
    return out


def all_frames(F, off, rs, res, sch, variant="work") -> pd.DataFrame:
    fr = []
    for r in rs:
        rr = res[(sch, r.design, r.lib, r.mono, variant)]
        fr += [pb.frame(F, r, off[r.design], rr, k) for k in ("a", "b")]
    out = pd.concat(fr)
    out["model"] = out["model"] + ("__" + sch if variant == "work" else "__" + sch + "__" + variant)
    return out


def b7_rows(F, grid, fc, p1, p2, sch):
    """Б-7: защита поверх xgb_mse_nohar_mono_b и поверх лучшей из Б-1…Б-5 (по QLIKE(all) этого расписания, пометка)."""
    h = p1[p1.model == "harj_b"].sort_values("d").reset_index(drop=True)
    bst = p2[p2.model == BEST2].sort_values("d").reset_index(drop=True)
    q = {m: metrics.qlike(f["rv"], f["sigma2"]).mean() for m, f in fc.groupby("model") if m.endswith("_b__" + sch)}
    top = min(q, key=q.get)
    top_f = fc[fc.model == top].sort_values("d").reset_index(drop=True)
    rvw = np.exp(F.X[np.asarray(grid), 1])
    dfn = post.b7_defense(rvw)
    rows = []
    for name, base in (("b7_p2best", bst), ("b7_best_b1_b5[%s]" % top.split("__")[0], top_f)):
        o = base.copy()
        assert (o["d"].to_numpy() == h["d"].to_numpy()).all()
        o["sigma2"] = post.b7_sigma2(base["sigma2"].to_numpy(), h["sigma2"].to_numpy(), dfn)
        o["model"] = "%s_b__%s" % (name, sch)
        rows.append(o)
    share = pd.DataFrame({"year": F.dates[np.asarray(grid)].astype("datetime64[Y]").astype(int) + 1970, "defense": dfn})
    return pd.concat(rows), share.groupby("year")["defense"].agg(["mean", "sum", "size"]).reset_index().assign(schedule=sch)


def leak_table(F, off, rs, res, rv_m_q) -> pd.DataFrame:
    out = []
    for r in rs:
        row = {"model": r.name + "_b", "qlike_rv_m": rv_m_q}
        for v in ("work", "oracle", "perm", "full"):
            f = pb.frame(F, r, off[r.design], res[("monthly", r.design, r.lib, r.mono, v)], "b")
            row[v] = float(metrics.qlike(f["rv"], f["sigma2"]).mean())
        out.append(row)
    return pd.DataFrame(out)


def dm_table(allf: pd.DataFrame, h: pd.DataFrame, names: list) -> pd.DataFrame:
    rows = []
    base = h.sort_values("d").reset_index(drop=True)
    for samp, mk in (("all", lambda y: y > 0), ("<=2021", lambda y: y <= 2021), ("2022+", lambda y: y >= 2022)):
        m = mk(base["target_date"].dt.year).to_numpy()
        part = []
        for n in names:
            f = allf[allf.model == n].sort_values("d").reset_index(drop=True)
            part.append({"model": n, "sample": samp, **dm.dm(f["rv"][m], base["sigma2"][m], f["sigma2"][m])})
        t = pd.DataFrame(part)
        t["p_holm"] = dm.holm(t["p"].to_numpy())
        rows.append(t)
    return pd.concat(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    F = ft.build(pd.read_parquet(prov.BARS))
    grid = har.eligible(har.Panel(F.dates, F.seg, F.rv, F.X[:, :4]))
    p1 = pd.read_parquet(OUT / "forecasts_p1.parquet")
    p2 = {"daily": pd.read_parquet(OUT / "forecasts_p2.parquet"), "monthly": pd.read_parquet(OUT / "forecasts_p2_monthly.parquet")}
    D, off = pb.build_designs(F)
    rs = pb.rows()
    res = run_fits(F, D, grid, params_from_p2(), rs)
    rvi, h = pd.read_parquet(prov.RVI), p1[p1.model == "harj_b"]
    parts, shares = [], []
    for sch, _ in SCHED:
        fc = all_frames(F, off, rs, res, sch)
        b7, sh = b7_rows(F, grid, fc, p1, p2[sch], sch)
        parts += [fc, b7]
        shares.append(sh)
    allf = pd.concat(parts)
    allf.to_parquet(OUT / "forecasts_pb.parquet")
    refs = pd.concat([h.assign(model="harj_b_p1")] + [p2[s][p2[s].model == BEST2].assign(model="%s__p2_%s" % (BEST2, s)) for s, _ in SCHED])
    mt = metrics.metrics_table(pd.concat([allf, refs]), rvi)
    mt = pd.concat([mt, pb.extra_slices(pd.concat([allf, refs]))])
    prov3.write_csv(mt, TAB / "pb_all.csv", "Б-1…Б-7 (Ж-198); срезы этапа 1 + no2022; schedule в имени модели")
    prov3.write_csv(pd.concat(shares), TAB / "pb_defense.csv", "доля дней защиты Б-7 по годам даты d")
    rm = p1[p1.model == "rv_m"]
    prov3.write_csv(leak_table(F, off, rs, res, metrics.summary(rm["rv"], rm["sigma2"])["qlike"]), TAB / "pb_leak.csv",
                    "(б); work/oracle/perm — monthly, full — одна модель на всей выборке")
    write_verdict(mt, allf, h)


def write_verdict(mt, allf, h):
    q2 = pd.read_csv(TAB / "p2_metrics.csv", comment="#")
    q2 = q2[q2.model.str.endswith("_b") & ~q2.model.str.endswith("_btr")].pivot(index="model", columns="slice", values="qlike").min()
    ref = mt[mt.model == "harj_b_p1"].set_index("slice")["qlike"]
    sel = mt[mt.model.str.contains("_b__") & ~mt.model.str.contains("__p2_")]
    prov3.write_csv(pb.verdict(sel, ref, q2), TAB / "pb_verdict.csv", "критерий GB: closes2022 / strong / weak")
    top = sel[sel.slice == "all"].sort_values("qlike")["model"].head(3).tolist()
    prov3.write_csv(dm_table(allf, h, top), TAB / "pb_dm.csv", "DM к harj_b, NW лаг 5; 3 лучших по QLIKE(all) среди Б")
    print("готово; лучшие 3:", top)


if __name__ == "__main__":
    main()
