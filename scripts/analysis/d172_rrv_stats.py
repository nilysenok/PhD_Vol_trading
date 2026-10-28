"""Д17.2: сводка RRV против RV (тот же тикер-день, окно и карантин) -> docs/otchety/d172/korr.csv."""
from pathlib import Path

import numpy as np
import pandas as pd

MAIN = Path("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading")
ROOT = Path(__file__).resolve().parents[2]
r = pd.read_parquet(MAIN / "data/iss_check/rrv_1850.parquet")
v = pd.read_parquet(MAIN / "data/iss_check/rv_1850_noquar_28.parquet")
r["date"], v["date"] = pd.to_datetime(r.date), pd.to_datetime(v.date)
m = r.merge(v[["date", "ticker", "rv", "is_main17"]], on=["date", "ticker"])
m = m[m.date <= "2026-02-03"]
rows = []
for t, g in m.groupby("ticker"):
    ok = (g.rv > 0) & (g.rrv_pk > 0) & (g.rrv_gk > 0)
    lg = np.log(g[ok][["rv", "rrv_pk", "rrv_gk"]])
    rows.append({"ticker": t, "main17": bool(g.is_main17.iloc[0]), "days": len(g),
                 "zero_rrv_days": int((g.rrv_pk == 0).sum()),
                 "corr_ln_pk": lg.rv.corr(lg.rrv_pk), "corr_ln_gk": lg.rv.corr(lg.rrv_gk),
                 "corr_lvl_pk": g.rv.corr(g.rrv_pk), "corr_lvl_gk": g.rv.corr(g.rrv_gk),
                 "corr_ln_pk_gk": lg.rrv_pk.corr(lg.rrv_gk),
                 "med_pk_rv": float((g.rrv_pk / g.rv)[ok].median()), "med_gk_rv": float((g.rrv_gk / g.rv)[ok].median())})
k = pd.DataFrame(rows).sort_values(["main17", "ticker"], ascending=[False, True])
out = ROOT / "docs/otchety/d172/korr.csv"
k.round(4).to_csv(out, index=False)
pd.set_option("display.width", 250)
print(k.round(3).to_string(index=False))
for f, g in k.groupby("main17"):
    print("main17" if f else "extra11", g[["corr_ln_pk", "corr_ln_gk", "corr_lvl_pk", "corr_lvl_gk", "med_pk_rv", "med_gk_rv"]].median().round(3).to_dict(),
          "zero days", int(g.zero_rrv_days.sum()))
x = pd.read_parquet(MAIN / "data/regen/features_pit_v3_rrv.parquet")
for c in ["rrv_pk_target_h1", "rrv_gk_target_h1", "rrv_pk_target_mean_h22"]:
    print(c, "нулей", int((x[c] == 0).sum()), "NaN", int(x[c].isna().sum()), "| rv_target_h1 NaN", int(x.rv_target_h1.isna().sum()))
print("corr ln rv_target_h1 vs ln rrv_pk_target_h1 (17):",
      round(np.log(x.rv_target_h1).corr(np.log(x.rrv_pk_target_h1.where(x.rrv_pk_target_h1 > 0))), 4))
print("corr rrv_pk_d vs ln rv (17):", round(x.rrv_pk_d.corr(np.log(x.rv.where(x.rv > 0))), 4))
