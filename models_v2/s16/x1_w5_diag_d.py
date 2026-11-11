"""W5-A: блок D `W5_DIAG.csv` (перенос `w5/robust.py`, логика без изменений) — ПОСЛЕ ПРОСМОТРА блоков A–C (вне критерия, только устойчивость найденного состояния gap). Только чтение рядов.
Публичное API: `gain(rv, f, g)` -> (выигрыш группового множителя к общему в % QLIKE, словарь множителей по группам); `rows_d(x, st)` -> список dict-строк блока `D_gap_posthoc`:
по моделям X1, M, K2, K1 и окнам (годы, h1, h2, all, без 1 % крупнейших потерь X1, без 2018-04-09 и 2022-02-21…2022-04-01, h2 без 2022); по Δ_d (0, 1, 2, ≥ 3); gap × e;
доля совпадений меток gap при сдвиге на 252 строки."""
from __future__ import annotations

import numpy as np

from models_v2.s16 import x1_w3_core as C
from models_v2.s16.x1_w5_diag_abc import PERS


def gain(rv, f, g):
    """(QLIKE группового множителя / QLIKE общего множителя − 1)·100 и множители по группам."""
    c0 = (rv / f).mean()
    cg = {k: (rv[g == k] / f[g == k]).mean() for k in np.unique(g)}
    fg = f * np.array([cg[k] for k in g])
    return 100 * (C.qlike(rv, fg).mean() / C.qlike(rv, f * c0).mean() - 1), cg


def rows_d(x, st):  # xtask: длинная — перенос robust.rows_d без изменения логики
    rows = []
    rv = x["rv"].to_numpy()
    yr, half = x["yr"].to_numpy(), x["half"].to_numpy()
    gap = st["gap"]
    delta = ((x["t_plus"] - x["d"]).dt.days - 1).to_numpy()
    dcat = np.where(delta == 0, "0", np.where(delta == 1, "1", np.where(delta == 2, "2", ">=3")))
    lx = C.qlike(rv, x["F"].to_numpy())
    top = lx >= np.quantile(lx, 0.99)
    dd = x["t_plus"]
    shock = ((dd == "2018-04-09") | ((dd >= "2022-02-21") & (dd <= "2022-04-01"))).to_numpy()
    wins = [(y, yr == y) for y in PERS] + [("h1", half == "h1"), ("h2", half == "h2"), ("all", np.ones(len(x), bool)),
                                           ("all_no_top1pct", ~top), ("all_no_shocks", ~shock), ("h2_no2022", (half == "h2") & (yr != "2022"))]
    for mname in ("X1", "M", "K2", "K1"):
        f = x["F" if mname == "X1" else mname].to_numpy()
        u = np.log(rv / f)
        for w, m in wins:
            gr, cg = gain(rv[m], f[m], gap[m])
            a, b = m & (gap == "1"), m & (gap == "0")
            rows.append(dict(block="D_gap_posthoc", state="gap", group="model=" + mname, window=w, n=int(m.sum()), n_gap=int(a.sum()),
                             c_gap=cg.get("1", np.nan), c_nogap=cg.get("0", np.nan), ratio_c=cg.get("1", np.nan) / cg.get("0", np.nan),
                             gain_vs_common_pct=gr, mean_u_diff=u[a].mean() - u[b].mean(), median_u_diff=np.median(u[a]) - np.median(u[b]),
                             var_u_gap=u[a].var(ddof=1), var_u_nogap=u[b].var(ddof=1)))
    f = x["F"].to_numpy()
    for w, m in (("h1", half == "h1"), ("h2", half == "h2"), ("h2_no2022", (half == "h2") & (yr != "2022")), ("all", np.ones(len(x), bool))):
        for k in ("0", "1", "2", ">=3"):
            mm = m & (dcat == k)
            rows.append(dict(block="D_gap_posthoc", state="delta_cat", group=k, window=w, n=int(mm.sum()),
                             c_gap=(rv[mm] / f[mm]).mean() if mm.sum() else np.nan, mean_u_diff=np.log(rv[mm] / f[mm]).mean() if mm.sum() else np.nan))
        for e in (0, 1):
            for gk in ("0", "1"):
                mm = m & (x["s"].to_numpy() == e) & (gap == gk)
                rows.append(dict(block="D_gap_posthoc", state="gap_x_e", group="e=%d,gap=%s" % (e, gk), window=w, n=int(mm.sum()),
                                 c_gap=(rv[mm] / f[mm]).mean() if mm.sum() else np.nan, mean_u_diff=np.log(rv[mm] / f[mm]).mean() if mm.sum() else np.nan))
    sh = np.roll(gap, 252)
    rows.append(dict(block="D_gap_posthoc", state="gap_shift252", group="share_equal", window="all", n=len(gap),
                     c_gap=float((sh == gap).mean()), mean_u_diff=float(((sh == "1") & (gap == "1")).sum() / (gap == "1").sum())))
    return rows
