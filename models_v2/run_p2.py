"""Этап 2 (п. 2 блока Ж-197): подбор (один раз) -> счёт 16 строк с ежемесячным переобучением -> таблицы.

Запуск: python3 models_v2/run_p2.py   (процессов <= 6)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models_v2 import har, metrics, prov  # noqa: E402
from models_v2.s2 import boost, pool, prov2, tune  # noqa: E402
from models_v2.s2 import features as ft  # noqa: E402

OUT = prov.ROOT / "data/regen/v2_sber_h1"
TAB = prov.ROOT / "docs/tables/v2_sber_h1"
VARIANTS = ("work", "oracle", "perm")


def do_tuning(F):
    T, dt = tune.tune_setup(F)
    jobs = [((lib, loss, dp, dl), boost.Spec(lib, loss, False, False), dp, dl, T, dt, "work")
            for lib, loss, dp, dl in tune.candidates()]
    rows = []
    for (lib, loss, dp, dl), r in pool.map_jobs(F, jobs):
        rows.append({"lib": lib, "loss": loss, "depth": dp, "delta": dl, "qlike_b": tune.score(F, dt, r),
                     "n_trees": r["k"], "s2_es": r["s2_es"], "n_train": r["n_train"], "n_val": len(dt)})
    df = pd.DataFrame(rows).sort_values(["lib", "loss", "depth", "delta"]).reset_index(drop=True)
    params = tune.choose(df)
    df["chosen"] = [params[(r.lib, r.loss)] == (r.depth, r.delta) for r in df.itertuples()]
    prov2.write_csv(df, TAB / "p2_tuning.csv", "подбор: обучение цели<=2015-12-31, проверка цели 2016, QLIKE(б)")
    return params


def collect(F, grid, params, daily=True, variants=VARIANTS, with_full=True):
    bl = boost.blocks(F, grid, daily)
    jobs, dmap = [], {}
    for spec in boost.all_specs():
        dp, dl = params[(spec.lib, spec.loss)]
        for T, ds in bl.items():
            for v in variants:
                key = (spec.name, v, T)
                jobs.append((key, spec, dp, dl, T, ds, v))
                dmap[key] = ds
        if with_full:
            key = (spec.name, "full", -1)
            jobs.append((key, spec, dp, dl, len(F.dates) - 1, [int(x) for x in grid], "full"))
            dmap[key] = [int(x) for x in grid]
    t0 = time.time()
    res = pool.map_jobs(F, jobs)
    print("счёт (%s): %d задач, %.1f мин" % ("ежедневно" if daily else "ежемесячно", len(jobs), (time.time() - t0) / 60))
    return res, dmap


def assemble(F, res, dmap):
    ser, log, imp = {}, [], {}
    for (name, v, T), r in res:
        ds = np.array(dmap[(name, v, T)])
        ser.setdefault((name, v), []).append(pd.DataFrame(
            {"i": ds, "f": r["f"], "s2_es": r["s2_es"], "s2_tr": r["s2_tr"]}))
        if v == "work":
            log.append({"model": name, "T": T, "date_T": F.dates[T], "n_trees": r["k"], "n_train": r["n_train"],
                        "no_split": r["no_split"], "s2_es": r["s2_es"], "s2_tr": r["s2_tr"], "sec": r["sec"]})
            if not r["no_split"]:
                imp.setdefault(name, []).append(r["imp"])
    return {k: pd.concat(v).sort_values("i").reset_index(drop=True) for k, v in ser.items()}, pd.DataFrame(log), imp


def forecast_rows(F, s, name, kind):
    s2 = {"a": 0.0, "b": s["s2_es"] / 2, "btr": s["s2_tr"] / 2}[kind]
    i = s["i"].to_numpy()
    return pd.DataFrame({"d": F.dates[i], "target_date": F.dates[i + 1], "model": "%s_%s" % (name, kind),
                         "sigma2": np.exp(s["f"].to_numpy() + s2), "rv": F.rv[i + 1]})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    F = ft.build(pd.read_parquet(prov.BARS))
    grid = har.eligible(har.Panel(F.dates, F.seg, F.rv, F.X[:, :4]))
    p1 = pd.read_parquet(OUT / "forecasts_p1.parquet")
    d1 = np.sort(p1["d"].unique().astype("datetime64[ns]"))
    assert np.array_equal(d1, np.sort(F.dates[grid].astype("datetime64[ns]"))), "сетка должна совпасть с этапом 1"
    params = do_tuning(F)
    print("параметры:", params)
    names = [s.name for s in boost.all_specs()]
    rvi = pd.read_parquet(prov.RVI)
    ref = p1[p1.model == "harj_b"].assign(model="harj_b_p1")
    # чувствительность: ежемесячное переобучение (только рабочий вариант)
    res, dmap = collect(F, grid, params, daily=False, variants=("work",), with_full=False)
    ser, log, _ = assemble(F, res, dmap)
    fcm = pd.concat([forecast_rows(F, ser[(n, "work")], n, k) for n in names for k in ("a", "b", "btr")])
    fcm.to_parquet(OUT / "forecasts_p2_monthly.parquet")
    prov2.write_csv(metrics.metrics_table(pd.concat([fcm, ref]), rvi), TAB / "p2_metrics_monthly.csv",
                    "чувствительность: переобучение раз в месяц")
    # основная версия: ежедневное переобучение (Б4: замер уложился в минуты)
    res, dmap = collect(F, grid, params, daily=True)
    ser, log, imp = assemble(F, res, dmap)
    fc = pd.concat([forecast_rows(F, ser[(n, "work")], n, k) for n in names for k in ("a", "b", "btr")])
    fc.to_parquet(OUT / "forecasts_p2.parquet")
    log.to_parquet(OUT / "retrain_p2.parquet")
    prov2.write_csv(metrics.metrics_table(pd.concat([fc, ref]), rvi), TAB / "p2_metrics.csv")
    write_side_tables(F, ser, log, imp, params, names, ref, fc)


def write_side_tables(F, ser, log, imp, params, names, ref, fc):
    spec = {s.name: s for s in boost.all_specs()}
    rows = []
    for n in names:
        cols = [ft.NAMES[j] for j in spec[n].cols]
        m = np.mean(imp[n], axis=0)
        rows += [{"model": n, "feature": c, "gain_share": float(g)} for c, g in zip(cols, m)]
    prov2.write_csv(pd.DataFrame(rows), TAB / "p2_importance.csv", "gain нормирован, среднее по переобучениям")
    lk = []
    for n in names:
        row = {"model": n}
        for v in ("work", "oracle", "perm", "full"):
            s = ser[(n, v)]
            s2 = np.exp(s["f"] + s["s2_es"] / 2)
            row[v] = float(metrics.qlike(F.rv[s["i"] + 1], s2).mean())
        lk.append(row)
    p1 = pd.read_parquet(OUT / "forecasts_p1.parquet")
    r = p1[p1.model == "rv_m"]
    lkd = pd.DataFrame(lk).assign(qlike_rv_m=metrics.summary(r["rv"], r["sigma2"])["qlike"])
    prov2.write_csv(lkd, TAB / "p2_leak.csv", "(б); work/oracle/perm — ежедневное переобучение, full — одна модель на всей выборке")
    g = log.groupby("model").agg(n_retrain=("T", "size"), mean_trees=("n_trees", "mean"), mean_sec=("sec", "mean"),
                                 n_no_split=("no_split", "sum"), min_n_train=("n_train", "min"))
    g["daily_est_min_6proc"] = g["mean_sec"] * len(ref) / 6 / 60
    prov2.write_csv(g.reset_index(), TAB / "p2_retrain.csv", "ежедневное переобучение; no_split — модель без разбиений")
    print("готово; параметры", params)


if __name__ == "__main__":
    main()
