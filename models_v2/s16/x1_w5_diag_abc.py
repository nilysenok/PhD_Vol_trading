"""W5-A: блоки A–C диагностики `W5_DIAG.csv` (перенос `w5/diag.py`, логика без изменений). Только чтение рядов; обучения нет.
Публичное API: константы `PERS` (2018…2026), `GRID` (π: 0…1 шаг 0,05), `MINPAST` (60); `block_a(x, st)` — где X1 проигрывает (по группам и окнам); `block_b(x, st)` — перенос между
половинами (B_level: множитель уровня; B_trust: постоянный π для состояний sgnCM, tCM, absCM, tK1M, pi, e); `block_c(x, st)` — смещение ln(RV/F) в днях события. Каждая -> список dict-строк.
`x` — окно M из `x1_w5_diag.build()`, `st` — словарь состояний (строковые группы, `na` — меньше 60 прошлых значений)."""
from __future__ import annotations

import numpy as np

from models_v2.s16 import x1_w3_core as C

GRID = np.round(np.arange(21) * 0.05, 2)
PERS = [str(y) for y in range(2018, 2027)]
MINPAST = 60


def block_a(x, st):
    rows = []
    qK2 = x["l_K2"].mean()
    n = len(x)
    DK2, DM = (x["l_X1"] - x["l_K2"]).to_numpy(), (x["l_X1"] - x["l_M"]).to_numpy()
    yr, half = x["yr"].to_numpy(), x["half"].to_numpy()
    for s, g in st.items():
        for grp in sorted(set(g)):
            m = g == grp
            sgn_all = np.sign(DK2[m].mean())
            same = sum(np.sign(DK2[m & (yr == y)].mean()) == sgn_all for y in PERS if (m & (yr == y)).sum() > 0)
            nper = sum((m & (yr == y)).sum() > 0 for y in PERS)
            for w, mw in (("all", np.ones(n, bool)), ("h1", half == "h1"), ("h2", half == "h2"), ("h2_no2022", (half == "h2") & (yr != "2022"))):
                mm = m & mw
                if mm.sum() == 0:
                    continue
                rows.append(dict(block="A_loss", state=s, group=grp, window=w, n=int(mm.sum()), share=mm.sum() / mw.sum(),
                                 mean_D_M=DM[mm].mean(), mean_D_K2=DK2[mm].mean(),
                                 contrib_K2_pp=100 * DK2[mm].sum() / (n * qK2) if w == "all" else 100 * DK2[mm].sum() / (mw.sum() * x.loc[mw, "l_K2"].mean()),
                                 mean_lnRVF_X1=np.log(x.loc[mm, "rv"] / x.loc[mm, "F"]).mean(),
                                 years_same_sign=same if w == "all" else np.nan, n_periods=nper if w == "all" else np.nan))
    return rows


def _q(rv, f):
    return C.qlike(rv, f).mean()


def block_b(x, st):
    rows = []
    rv, F, M, Cc = x["rv"].to_numpy(), x["F"].to_numpy(), x["M"].to_numpy(), x["C"].to_numpy()
    half, yr = x["half"].to_numpy(), x["yr"].to_numpy()
    H = {"h1": half == "h1", "h2": half == "h2"}
    trust_states = ("sgnCM", "tCM", "absCM", "tK1M", "pi", "e")
    for s, g in st.items():
        ok = g != "na"
        for kind in ("level", "trust"):
            if kind == "trust" and s not in trust_states:
                continue
            fit = {}
            for hn, hm in H.items():
                mm = hm & ok
                if kind == "level":
                    c0 = (rv[mm] / F[mm]).mean()
                    cg = {grp: (rv[mm & (g == grp)] / F[mm & (g == grp)]).mean() for grp in set(g[mm])}
                    fit[hn] = (c0, cg)
                else:
                    L = np.column_stack([C.qlike(rv, C.combine(M, Cc, np.full(len(rv), p))) for p in GRID])
                    p0 = GRID[np.argmin(L[mm].mean(0))]
                    pg = {grp: GRID[np.argmin(L[mm & (g == grp)].mean(0))] for grp in set(g[mm])}
                    fit[hn] = (p0, pg)
            for src, dst, extra in (("h1", "h2", None), ("h2", "h1", None), ("h1", "h2", "no2022"), ("h1", "h1", "insample"), ("h2", "h2", "insample")):
                mm = H[dst] & ok
                if extra == "no2022":
                    mm = mm & (yr != "2022")
                a0, ag = fit[src]
                if kind == "level":
                    f_ctrl = F[mm] * a0
                    f_st = F[mm] * np.array([ag.get(gg, a0) for gg in g[mm]])
                else:
                    f_ctrl = C.combine(M[mm], Cc[mm], np.full(mm.sum(), a0))
                    f_st = C.combine(M[mm], Cc[mm], np.array([ag.get(gg, a0) for gg in g[mm]]))
                qx, qc, qs = _q(rv[mm], F[mm]), _q(rv[mm], f_ctrl), _q(rv[mm], f_st)
                rows.append(dict(block="B_" + kind, state=s, group="|".join("%s:%.3f" % (k, v) for k, v in sorted(ag.items())),
                                 window="%s->%s%s" % (src, dst, "" if extra is None else "_" + extra), n=int(mm.sum()),
                                 q_X1=qx, q_ctrl=qc, q_state=qs, d_state_vs_X1_pct=100 * (qs / qx - 1), d_state_vs_ctrl_pct=100 * (qs / qc - 1),
                                 d_ctrl_vs_X1_pct=100 * (qc / qx - 1), ctrl_param=a0))
    return rows


def block_c(x, st):
    rows = []
    yr = x["yr"].to_numpy()
    for s in ("ev_cbr", "ev_cpius", "ev_fomc", "ev_exp", "ev_rebal", "ev_any", "gap"):
        g = st[s]
        for y in PERS + ["h1", "h2", "all"]:
            my = (x["half"].to_numpy() == y) if y in ("h1", "h2") else (np.ones(len(x), bool) if y == "all" else yr == y)
            r = dict(block="C_event", state=s, group="1-0", window=y, n=int((my & (g == "1")).sum()))
            for k in ("X1", "K1", "K2", "M"):
                f = x["F" if k == "X1" else k].to_numpy()
                u = np.log(x["rv"].to_numpy() / f)
                a, b = my & (g == "1"), my & (g == "0")
                r["bias_diff_" + k] = u[a].mean() - u[b].mean() if a.sum() else np.nan
                r["qlike_ratio_" + k] = C.qlike(x["rv"].to_numpy()[a], f[a]).mean() / C.qlike(x["rv"].to_numpy()[b], f[b]).mean() if a.sum() else np.nan
            rows.append(r)
    return rows
