"""W8-A: блоки C_group, D_transfer, E_verdict из `w8/build.py` БЕЗ изменения логики (критерий П1–П5 — `W8_SPEC.md` разд. 1.0). Только чтение.
Публичное API:
- `halves(x)`: маски h1 (t⁺ 2018–2021), h2 (2022–2026), h2no22.
- `fit_groups(x, m, grp) -> (β групп, общий β, n групп)`: оценщик `beta_est` на половине m по дням gap = 1; группа < 15 дней или `na` — без своего β (берёт общий).
- `apply_q(x, m, grp, betas, pooled) -> (Q признака, Q контроля, Q X1)` на половине m.
- `blocks_cde(x, feats) -> (строки C_group, D_transfer, E_verdict)`; E_verdict: П1…П5 и passed (все пять)."""
from __future__ import annotations

import numpy as np

from models_v2.s16 import x1_w8_diag_core as K


def halves(x):
    h1 = (x["half"] == "h1").to_numpy()
    h2 = (x["half"] == "h2").to_numpy()
    h2n = h2 & (x["yr"] != "2022").to_numpy()
    return {"h1": h1, "h2": h2, "h2no22": h2n}


def fit_groups(x, m, grp):
    """β^A по группам и общий β^A на половине m. -> (dict групп, общий β, n групп)."""
    u, g = x["u"].to_numpy(), x["gap"].to_numpy()
    u0 = u[m & (g == 0)]
    pooled = K.beta_est(u[m & (g == 1)], u0)
    out, ns = {}, {}
    for G in sorted(set(grp[m & (g == 1)])):
        s = m & (g == 1) & (grp == G)
        ns[G] = int(s.sum())
        if G == "na" or s.sum() < K.MIN_GROUP:
            continue
        out[G] = K.beta_est(u[s], u0)
    return out, pooled, ns


def apply_q(x, m, grp, betas, pooled):
    u, g = x["u"].to_numpy(), x["gap"].to_numpy()
    c = np.array([betas.get(G, pooled) for G in grp]) * g
    cp = pooled * g
    lf = np.exp(u - c) - (u - c) - 1
    lc = np.exp(u - cp) - (u - cp) - 1
    l0 = x["l_X1"].to_numpy()
    return float(lf[m].mean()), float(lc[m].mean()), float(l0[m].mean())


def blocks_cde(x, feats):  # xtask: длинная — перенос w8/build.py без изменения логики
    H = halves(x)
    u, g = x["u"].to_numpy(), x["gap"].to_numpy()
    dlw = (x["l_W5G"] - x["l_X1"]).to_numpy()
    rows_c, rows_d, rows_e = [], [], []
    dirs = [("h1->h2", "h1", "h2"), ("h2->h1", "h2", "h1"), ("h1->h2no22", "h1", "h2no22"), ("h2no22->h1", "h2no22", "h1")]
    for fname, grp in feats.items():
        grp = np.asarray(grp, object)
        cstar = {}
        for hn, hm in H.items():
            Qx = float(x.loc[hm, "l_X1"].mean())
            u0 = u[hm & (g == 0)]
            cstar[hn] = {}
            for G in sorted(set(grp[hm & (g == 1)])):
                s = hm & (g == 1) & (grp == G)
                cs = float(np.log(np.mean(np.exp(u[s]))))
                be = K.beta_est(u[s], u0) if s.sum() >= 2 else np.nan
                if s.sum() >= K.MIN_GROUP and G != "na":
                    cstar[hn][G] = cs
                rows_c.append(dict(block="C_group", feature=fname, half=hn, group=G, n=int(s.sum()), beta_expost=cs, beta_est=be,
                                   beta_app=float(x["c_app"].to_numpy()[s].mean()), mean_u=float(u[s].mean()),
                                   delta_W5G_pct_in_group=float(dlw[s].mean() / x["l_X1"].to_numpy()[s].mean() * 100),
                                   contrib_pp_half=float(dlw[s].sum() / (hm.sum() * Qx) * 100), share_help=float((dlw[s] < 0).mean())))
        res = {}
        for dn, a, b in dirs:
            betas, pooled, ns = fit_groups(x, H[a], grp)
            qf, qc, q0 = apply_q(x, H[b], grp, betas, pooled)
            res[dn] = dict(d_ctrl=(qf - qc) / qc * 100, d_x1=(qf - q0) / q0 * 100)
            rows_d.append(dict(block="D_transfer", feature=fname, direction=dn, Q_X1=q0, Q_ctrl=qc, Q_feat=qf, d_feat_vs_ctrl_pct=res[dn]["d_ctrl"],
                               d_feat_vs_X1_pct=res[dn]["d_x1"], d_ctrl_vs_X1_pct=(qc - q0) / q0 * 100, beta_pooled=pooled,
                               betas=";".join("%s=%.4f" % kv for kv in sorted(betas.items())), n_groups=";".join("%s=%d" % kv for kv in sorted(ns.items()))))
        common = sorted(set(cstar["h1"]) & set(cstar["h2no22"]))
        o1 = sorted(common, key=lambda G: cstar["h1"][G])
        o2 = sorted(common, key=lambda G: cstar["h2no22"][G])
        p1 = res["h1->h2"]["d_ctrl"] < 0 and res["h2->h1"]["d_ctrl"] < 0
        p2 = res["h1->h2"]["d_x1"] < 0 and res["h2->h1"]["d_x1"] < 0
        p3 = res["h1->h2no22"]["d_ctrl"] < 0 and res["h2no22->h1"]["d_ctrl"] < 0
        p4 = len(common) >= 2 and o1 == o2
        p5 = res["h1->h2no22"]["d_x1"] <= 0 and res["h2no22->h1"]["d_x1"] <= 0
        rows_e.append(dict(block="E_verdict", feature=fname, P1=int(p1), P2=int(p2), P3=int(p3), P4=int(p4), P5=int(p5), passed=int(p1 and p2 and p3 and p4 and p5),
                           order_h1="<".join(o1), order_h2no22="<".join(o2)))
    return rows_c, rows_d, rows_e
