"""Сравнение С9: σ_pred h=1 final2 → final3 (D22_h1). Только таблицы, без пересчёта."""
import sys
import numpy as np
import pandas as pd

T = "docs/tables/strat_dev/"
Y = ["2018", "2019", "2020", "2021"]


def rd(nm, new):
    p = T + nm + ("_final3.csv" if new else ".csv")
    return pd.read_csv(p, sep=";", decimal=",", encoding="utf-8-sig" if new else "utf-8")


def key(df, cols):
    return df[cols].astype(str).agg("|".join, axis=1)


rows = []


def add(block, o, n, design_fn, closed_col="closed_dev"):
    for (_, a), (_, b) in zip(o.iterrows(), n.iterrows()):
        r = dict(blok=block, dizayn=design_fn(a), h=a.get("h", "h1"))
        for c in Y:
            r[f"{c}_staryy"], r[f"{c}_novyy"] = a.get(c), b.get(c)
        r["srednee_po_godam_staryy"] = np.mean([a.get(c) for c in Y]) if pd.notna(a.get("2018")) else np.nan
        r["srednee_po_godam_novyy"] = np.mean([b.get(c) for c in Y]) if pd.notna(b.get("2018")) else np.nan
        r["period_staryy"], r["period_novyy"] = a.get("sharpe"), b.get("sharpe")
        r["d_period"] = b.get("sharpe") - a.get("sharpe") if pd.notna(a.get("sharpe")) else np.nan
        r["zakryto_staryy"], r["zakryto_novyy"] = a.get(closed_col, np.nan), b.get(closed_col, np.nan)
        r["ves_staryy"], r["ves_novyy"] = a.get("mean_weight", np.nan), b.get("mean_weight", np.nan)
        r["d_k_A_staryy"], r["p_k_A_staryy"] = a.get("d_vs_A", np.nan), a.get("p_vs_A", np.nan)
        r["d_k_A_novyy"], r["p_k_A_novyy"] = b.get("d_vs_A", np.nan), b.get("p_vs_A", np.nan)
        r["S8_usloviy_staryy"], r["S8_usloviy_novyy"] = a.get("n_met", np.nan), b.get("n_met", np.nan)
        r["S8_staryy"], r["S8_novyy"] = a.get("passed", np.nan), b.get("passed", np.nan)
        r["v_polose_staryy"] = a.get("screened_in", np.nan)
        r["v_polose_novyy"] = b.get("screened_in", np.nan)
        rows.append(r)


# A — эталон: строка «A (эталон)» из B-таблицы (спот) + С8 неприменим
bo, bn = rd("s9_B_targetirovanie", False), rd("s9_B_targetirovanie", True)
a_o = bo[(bo.family == "A (эталон)") & (bo.scenario == "спот")].rename(columns={"sharpe_net": "sharpe"})
a_n = bn[(bn.family == "A (эталон)") & (bn.scenario == "спот")].rename(columns={"sharpe_net": "sharpe"})
add("A", a_o, a_n, lambda a: "A основного дизайна (от σ не зависит)")
# D
do, dn = rd("s9_D_zatvor", False), rd("s9_D_zatvor", True)
assert (key(do, ["family", "point", "h"]) == key(dn, ["family", "point", "h"])).all()
m = do.h == "h1"
add("D", do[m], dn[m], lambda a: f"{a.family} · {a.point}")
# C
co, cn = rd("s9_C_rezhim", False), rd("s9_C_rezhim", True)
assert (key(co, ["family", "point", "sigma"]) == key(cn, ["family", "point", "sigma"])).all()
m = co.sigma == "hyb_h1"
add("C", co[m], cn[m], lambda a: f"{a.family} · {a.point}")
# B
m = (bo.family == "B") & (bo.scenario == "спот") & (bo.h == "h1")
add("B", bo[m].rename(columns={"sharpe_net": "sharpe"}), bn[m].rename(columns={"sharpe_net": "sharpe"}),
    lambda a: f"B · {a.point}")
# МСП: P2
po, pn = rd("s9_P2_multistrategiya", False), rd("s9_P2_multistrategiya", True)
import json
O = "/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/fastbt/s9/"
ch_o = json.load(open(O + "run_log.json"))["choice"]
ch_n = json.load(open(O + "run_log_final3.json"))["choice"]
lab = {k: f" (выбор по С8: было {ch_o[k]}, стало {ch_n[k]})" for k in ch_o}
add("МСП P2", po, pn, lambda a: f"P2 · {a.portfolio}" + next(
    (v for k, v in lab.items() if a.portfolio == f"одиночная {k}"), ""))
# МСП: P1
qo, qn = rd("s9_P1_vesa_tikerov", False), rd("s9_P1_vesa_tikerov", True)
m = qo.scenario == "спот"
add("МСП P1", qo[m], qn[m], lambda a: f"P1 · {a.base} · {a.weights}")

out = pd.DataFrame(rows)
out.insert(0, "status", "предварительно")
out.to_csv(T + "s9_sravnenie_sigma_final3.csv", sep=";", decimal=",", index=False,
           float_format="%.4f", encoding="utf-8-sig")

# неизменность h=5 и прочих источников
chk = []
for nm, col, vals in (("s9_D_zatvor", "h", ["h5"]), ("s9_C_rezhim", "sigma", ["hyb_h5", "harj_h1", "naive_h1"])):
    o, n = rd(nm, False), rd(nm, True)
    mm = o[col].isin(vals)
    num = o.select_dtypes("number").columns.drop(["sharpe_C", "d_vs_C", "p_vs_C"], errors="ignore")
    chk.append(f"{nm} {vals}: max|Δ| {np.nanmax(np.abs(o.loc[mm, num].values - n.loc[mm, num].values)):.1e}")
o, n = rd("s9_D_zatvor_harj", False), rd("s9_D_zatvor_harj", True)
num = o.select_dtypes("number").columns.drop(["sharpe_C", "d_vs_C", "p_vs_C"], errors="ignore")
chk.append(f"s9_D_zatvor_harj: max|Δ| {np.nanmax(np.abs(o[num].values - n[num].values)):.1e}")
print("\n".join(chk))
