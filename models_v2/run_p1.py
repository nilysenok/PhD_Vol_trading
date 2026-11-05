"""Этап 1 (п. 1 блока Ж-197): дневной ряд, HAR-J/HAR/эталоны, метрики, коэффициенты, arch, утечки.

Запуск: python3 models_v2/run_p1.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models_v2 import arch_check, daily, har, leak, metrics, prov  # noqa: E402

OUT = prov.ROOT / "data/regen/v2_sber_h1"
TAB = prov.ROOT / "docs/tables/v2_sber_h1"
MODELS = {"har_a": (har.COLS_HAR, False), "har_b": (har.COLS_HAR, True),
          "harj_a": (har.COLS_HARJ, False), "harj_b": (har.COLS_HARJ, True)}


def coef_table(p, grid):
    d = pd.Series(p.dates[grid])
    last = grid[d.groupby(d.dt.year).transform("max").eq(d).to_numpy()]
    rows = []
    for i in last:
        for name, cols in (("har", har.COLS_HAR), ("harj", har.COLS_HARJ)):
            b, s2, n = har.fit_window(p, int(i), cols)
            row = {"d": p.dates[i], "model": name, "n": n, "s2": s2}
            row.update({k: np.nan for k in ("b0", "b_d", "b_w", "b_m", "b_j")})
            row.update(dict(zip(("b0", "b_d", "b_w", "b_m", "b_j"), b)))
            rows.append(row)
    return pd.DataFrame(rows)


def leak_table(p, grid, fc):
    qrv = metrics.summary(fc[fc.model == "rv_m"]["rv"], fc[fc.model == "rv_m"]["sigma2"])["qlike"]
    rows = []
    for name, (cols, bias) in MODELS.items():
        for var, fn in (("work", leak.work), ("oracle", leak.oracle), ("full", leak.full_sample),
                        ("perm", leak.permuted)):
            q = leak.qlike_variant(p, grid, cols, fn, bias)
            rows.append({"model": name, "variant": var, "qlike": q, "qlike_rv_m": qrv})
    for bias in (False, True):  # диагностика ПОСЛЕ ПРОСМОТРА, не гейт: безусловное расширяющееся среднее
        q = leak.qlike_variant(p, grid, har.COLS_HAR, leak.const_control, bias)
        rows.append({"model": "const_" + ("b" if bias else "a"), "variant": "ref_posthoc", "qlike": q,
                     "qlike_rv_m": qrv})
    return pd.DataFrame(rows)


def sens_table(daily_df, rvi):
    """Чувствительность (не основная): HAR-J, где ln(1+J) берётся от J в долях (J_%2 / 1e4); RV, RV_w, RV_m — в %^2."""
    d2 = daily_df.copy()
    d2["j"] = d2["j"] / daily.SCALE
    fc = har.forecast_table(har.make_panel(d2))
    fc = fc[fc.model.isin(["harj_a", "harj_b"])].assign(model=lambda x: x.model + "_J_v_dolyah")
    t = metrics.metrics_table(fc, rvi)
    return t[t["slice"].isin(["all", "y2022", "mar_jun_2020"])]


def weekend_kept(bars, daily_df):
    """Сб/Вс до 2025-03-01, оставленные в ряду: дата, баров основной сессии, первый/последний begin."""
    m = daily.main_session(bars)
    g = m.groupby(m["begin"].dt.normalize())["begin"].agg(["size", "min", "max"])
    g = g[(g.index.dayofweek >= 5) & g.index.isin(daily_df.index)]
    return pd.DataFrame({"date": g.index, "n_bars": g["size"].values,
                         "first": g["min"].dt.strftime("%H:%M").values, "last": g["max"].dt.strftime("%H:%M").values})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bars = pd.read_parquet(prov.BARS)
    dly, excl = daily.build_daily(bars)
    dly.to_parquet(OUT / "daily_p1.parquet")
    prov.write_csv(excl, TAB / "p1_excluded_days.csv")
    prov.write_csv(weekend_kept(bars, dly), TAB / "p1_weekend_kept.csv")
    p = har.make_panel(dly)
    grid = har.eligible(p)
    fc = har.forecast_table(p)
    fc.to_parquet(OUT / "forecasts_p1.parquet")
    (OUT / "forecasts_p1.meta.txt").write_text(prov.header(), encoding="utf-8")
    rvi = pd.read_parquet(prov.RVI)
    prov.write_csv(metrics.metrics_table(fc, rvi), TAB / "p1_metrics.csv")
    prov.write_csv(coef_table(p, grid), TAB / "p1_coef.csv")
    prov.write_csv(arch_check.check_dates(p, grid), TAB / "p1_arch_check.csv")
    prov.write_csv(leak_table(p, grid, fc), TAB / "p1_leak.csv")
    prov.write_csv(sens_table(dly, rvi), TAB / "p1_sens_j.csv", "справка, не гейт")
    print("дней в ряду", len(dly), "исключено", len(excl), "сетка", len(grid))


if __name__ == "__main__":
    main()
