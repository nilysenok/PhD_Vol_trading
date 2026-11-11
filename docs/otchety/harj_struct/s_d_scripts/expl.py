"""S-D (Ж-256): разбор S-R по сохранённым рядам и параметрам. Обучения нет: только чтение parquet, пересчёт слоёв X1/W5 прежним кодом над C при фиксированных θ.
Пишет docs/otchety/harj_struct/S_D_EXPLAIN.csv. Запуск из корня: PYTHONPATH=. python3 -W ignore <sp>/s_d/expl.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import recon as RC  # noqa: E402

from models_v2.s16 import x1_w3_core as C  # noqa: E402
from models_v2.s18 import a_feat as AF  # noqa: E402
from models_v2.s18 import cm_layers as L  # noqa: E402

R = C.ROOT / "data/regen/harj_struct"
OUT = C.ROOT / "docs/otchety/harj_struct/S_D_EXPLAIN.csv"
SER = {"S-CK": "CM/ck", "S-RCK": "R/rck", "S-RCKb": "R/rckb", "S-RCKc": "R/rckc", "S-A2": "A/a2", "S-RA2": "R/ra2", "S-RA2b": "R/ra2b", "S-RA2c": "R/ra2c", "S-A0": "A/a0", "S-A1": "A/a1"}
ROWS = []


def put(sec, ser, item, window, n, value, base=""):
    ROWS.append(dict(section=sec, series=ser, item=item, base=base, window=window, n=int(n), value=float(value)))


def ell(rv, F):
    r = np.asarray(rv, float) / np.asarray(F, float)
    return r - np.log(r) - 1.0


def dpct(la, lb, m=None):
    m = np.ones(len(la), bool) if m is None else m
    return 100.0 * (la[m].mean() - lb[m].mean()) / lb[m].mean()


def contrib(la, lb, m):
    return 100.0 * (la - lb)[m].sum() / (len(la) * lb.mean())


def load():
    x, gap, a = RC.arrays()
    w5 = pd.read_parquet(C.ROOT / "data/regen/harj_x1/W5/w5g.parquet")
    on = (w5["gap"].to_numpy() == 1) & w5["refit_date"].notna().to_numpy()
    bg = np.where(on, w5["beta"].to_numpy(float), 0.0)
    S = {k: pd.read_parquet(R / (f + ".parquet")) for k, f in SER.items()}
    for k, df in S.items():
        assert (pd.to_datetime(df["d"]).to_numpy() == x["d"].to_numpy()).all(), k
        df["Cc"] = df["C_A"] if "C_A" in df else df["C"]
        df["X1c"] = df["X1_R"] if "X1_R" in df else (df["X1_A"] if "X1_A" in df else df["X1_V"])
    return x, gap, a, w5, bg, S


def ffr(x, bg, Cc):
    pi = x["pi"].to_numpy(float)
    return np.exp((1.0 - pi) * np.log(x["M"].to_numpy(float)) + pi * np.log(np.asarray(Cc, float)) + bg)


def sec_table(x, w5, bg, S):
    rv, n = x["rv"].to_numpy(float), len(x)
    lw, lx, lc8 = ell(rv, w5["F"]), ell(rv, x["F"]), ell(rv, x["C"])
    put("A_level", "W5-G", "qlike_F", "D_M", n, lw.mean())
    put("A_level", "X1", "qlike_F", "D_M", n, lx.mean())
    put("A_level", "cfg8", "qlike_C", "D_M", n, lc8.mean())
    for k, df in S.items():
        lf, lfr = ell(rv, df["F"]), ell(rv, ffr(x, bg, df["Cc"]))
        put("A_level", k, "qlike_F", "D_M", n, lf.mean())
        put("A_level", k, "qlike_X1layer", "D_M", n, ell(rv, df["X1c"]).mean())
        put("A_level", k, "qlike_C", "D_M", n, ell(rv, df["Cc"]).mean())
        put("A_level", k, "dC_vs_cfg8_pct", "D_M", n, dpct(ell(rv, df["Cc"]), lc8), "cfg8")
        put("A_level", k, "dX1_vs_X1_pct", "D_M", n, dpct(ell(rv, df["X1c"]), lx), "X1")
        put("B_decomp", k, "total_F_vs_W5G", "D_M", n, dpct(lf, lw), "W5-G")
        put("B_decomp", k, "frozen_pi", "D_M", n, dpct(lfr, lw), "W5-G")
        put("B_decomp", k, "layer_reaction", "D_M", n, 100.0 * (lf.mean() - lfr.mean()) / lw.mean(), "W5-G")
        d = x["d"]
        for lo in ("2018-07-02", "2019-08-01", "2020-08-03"):
            m = (d >= lo).to_numpy()
            put("C_window", k, "F_vs_W5G", "d>=" + lo, m.sum(), dpct(lf, lw, m), "W5-G")
            put("C_window", k, "Ffr_vs_W5G", "d>=" + lo, m.sum(), dpct(lfr, lw, m), "W5-G")
    for new, old in (("S-RCK", "S-CK"), ("S-RCKb", "S-CK"), ("S-RCKc", "S-CK"), ("S-RA2", "S-A2"), ("S-RA2b", "S-A2"), ("S-RA2c", "S-A2"),
                     ("S-RCK", "S-RCKb"), ("S-RCK", "S-RCKc"), ("S-RA2", "S-RA2b"), ("S-RA2", "S-RA2c")):
        la, lb = ell(rv, S[new]["F"]), ell(rv, S[old]["F"])
        fa, fb = ell(rv, ffr(x, bg, S[new]["Cc"])), ell(rv, ffr(x, bg, S[old]["Cc"]))
        tag = "%s|%s" % (new, old)
        put("D_pair", tag, "F", "D_M", n, dpct(la, lb), old)
        put("D_pair", tag, "Ffr", "D_M", n, dpct(fa, fb), old + " (Ffr)")
        put("D_pair", tag, "C", "D_M", n, dpct(ell(rv, S[new]["Cc"]), ell(rv, S[old]["Cc"])), old + " (C)")
        k24 = (x["t_plus"] == "2022-02-24").to_numpy()
        put("D_pair", tag, "F_contrib_2022-02-24", "t+=2022-02-24", 1, contrib(la, lb, k24), old)
        put("D_pair", tag, "F_without_2022-02-24", "D_M без t+ 2022-02-24", (~k24).sum(), dpct(la, lb, ~k24), old)
        y22 = (x["t_plus"].dt.year == 2022).to_numpy()
        put("D_pair", tag, "F_without_2022", "D_M без t+ 2022", (~y22).sum(), dpct(la, lb, ~y22), old)
        for lo in ("2019-08-01", "2020-08-03"):
            m = (x["d"] >= lo).to_numpy()
            put("D_pair", tag, "F", "d>=" + lo, m.sum(), dpct(la, lb, m), old)
        for y in range(2018, 2027):
            m = (x["t_plus"].dt.year == y).to_numpy()
            put("D_pair", tag, "F_contrib_year", str(y), m.sum(), contrib(la, lb, m), old)


def sec_cdelta(x, S):
    """Уровень C против cfg8: распределение δ = ln(C/cfg8), уровень против разброса (как S-C-WHY H2)."""
    rv, c8 = x["rv"].to_numpy(float), x["C"].to_numpy(float)
    l8 = ell(rv, c8)
    for k, df in S.items():
        dl = np.log(df["Cc"].to_numpy(float) / c8)
        act = np.abs(dl) > 1e-12
        n = act.sum()
        put("E_C", k, "share_days_C_ne_cfg8", "D_M", len(x), act.mean())
        if n < 10:
            continue
        mu = dl[act].mean()
        put("E_C", k, "mean_lnC_cfg8", "active", n, mu)
        put("E_C", k, "sd_lnC_cfg8", "active", n, dl[act].std(ddof=1))
        for q in (0.05, 0.5, 0.95):
            put("E_C", k, "q%02d_lnC_cfg8" % int(q * 100), "active", n, np.quantile(dl[act], q))
        put("E_C", k, "share_neg", "active", n, (dl[act] < 0).mean())
        put("E_C", k, "corr_delta_ln_rv_cfg8", "active", n, np.corrcoef(dl[act], np.log(rv / c8)[act])[0, 1])
        lvl = np.where(act, c8 * np.exp(mu), c8)
        disp = np.where(act, c8 * np.exp(dl - mu), c8)
        put("E_C", k, "C_only_level_vs_cfg8", "D_M", len(x), dpct(ell(rv, lvl), l8), "cfg8")
        put("E_C", k, "C_only_dispersion_vs_cfg8", "D_M", len(x), dpct(ell(rv, disp), l8), "cfg8")
        put("E_C", k, "C_full_vs_cfg8", "D_M", len(x), dpct(ell(rv, df["Cc"]), l8), "cfg8")


def sec_months(x, w5, bg, S):
    """Вклад дней по типу месяца R: до включения, θ = 0 (λ* = 0), θ ≠ 0. Δ(F против W5-G), п. п. QLIKE W5-G; Σ = Δ."""
    rv, lw = x["rv"].to_numpy(float), ell(x["rv"], w5["F"])
    for k in ("S-RCK", "S-RCKb", "S-RCKc", "S-RA2", "S-RA2b", "S-RA2c"):
        df = S[k]
        lam, act = df["lam"].to_numpy(float), df["active"].to_numpy(bool)
        typ = {"pre": ~act, "theta0": act & (lam == 0), "theta_ne0": act & (lam > 0)}
        lf, lfr = ell(rv, df["F"]), ell(rv, ffr(x, bg, df["Cc"]))
        mon = df.assign(mo=pd.to_datetime(df["d"]).dt.to_period("M")).drop_duplicates("mo")
        for t, m in typ.items():
            put("F_months", k, "days_" + t, "D_M", m.sum(), m.sum())
            put("F_months", k, "months_" + t, "D_M", len(mon), int({"pre": (~mon["active"]).sum(), "theta0": (mon["active"] & (mon["lam"] == 0)).sum(), "theta_ne0": (mon["active"] & (mon["lam"] > 0)).sum()}[t]))
            put("F_months", k, "contrib_F_" + t, "D_M", m.sum(), contrib(lf, lw, m), "W5-G")
            put("F_months", k, "contrib_Ffr_" + t, "D_M", m.sum(), contrib(lfr, lw, m), "W5-G")
    for arch, ks in (("CK", ("S-RCK", "S-RCKb", "S-RCKc")), ("A2", ("S-RA2", "S-RA2b", "S-RA2c"))):
        mo = {k: S[k].assign(mo=pd.to_datetime(S[k]["d"]).dt.to_period("M")).drop_duplicates("mo").set_index("mo") for k in ks}
        act = mo[ks[0]]["active"].to_numpy(bool)
        z = {k: (mo[k]["lam"].to_numpy(float) == 0)[act] for k in ks}
        put("F_agree", arch, "months_active", "D_M", act.sum(), act.sum())
        for i in range(3):
            for j in range(i + 1, 3):
                put("F_agree", arch, "same_theta0_status_%s_%s" % (ks[i], ks[j]), "active months", act.sum(), (z[ks[i]] == z[ks[j]]).mean())
        lam = {k: mo[k]["lam"].to_numpy(float)[act] for k in ks}
        put("F_agree", arch, "same_lambda_R_Rb", "active months", act.sum(), (lam[ks[0]] == lam[ks[1]]).mean())
        put("F_agree", arch, "same_lambda_R_Rc", "active months", act.sum(), (lam[ks[0]] == lam[ks[2]]).mean())


def sec_ck_parts(x, gap, a, w5, bg, S):
    """Разложение S-RCK при фиксированных θ (не абляция: θ не переобучаются): память γ, уровень θ0, затухание шока θ1."""
    rv, c8, lw, l8 = x["rv"].to_numpy(float), a["c8"], ell(x["rv"], w5["F"]), ell(x["rv"], x["C"])
    keep = {"full": (0, 1, 2, 3, 4, 5), "level_only": (2, 3), "memory_only": (0, 1), "decay_only": (4, 5),
            "no_level": (0, 1, 4, 5), "no_memory": (2, 3, 4, 5), "no_decay": (0, 1, 2, 3)}
    for k in ("S-RCK", "S-RCKb", "S-RCKc", "S-CK"):
        th = S[k][list(RC.TH)].to_numpy(float)
        for part, idx in keep.items():
            t = np.zeros_like(th)
            t[:, list(idx)] = th[:, list(idx)]
            Cp = c8 * RC.mult(a, t)
            lay = L.layers(x, Cp, gap)
            put("G_ck_parts", k, part + "|C_vs_cfg8", "D_M", len(x), dpct(ell(rv, Cp), l8), "cfg8")
            put("G_ck_parts", k, part + "|Ffr_vs_W5G", "D_M", len(x), dpct(ell(rv, ffr(x, bg, Cp)), lw), "W5-G")
            put("G_ck_parts", k, part + "|F_vs_W5G", "D_M", len(x), dpct(ell(rv, lay["F"]), lw), "W5-G")
            put("G_ck_parts", k, part + "|X1_vs_X1", "D_M", len(x), dpct(ell(rv, lay["X1_V"]), ell(rv, x["F"])), "X1")
        mon = S[k].assign(mo=pd.to_datetime(S[k]["d"]).dt.to_period("M")).drop_duplicates("mo")
        mon = mon[(mon[list(RC.TH)].abs().sum(axis=1) > 0)]
        put("G_ck_parts", k, "months_theta_ne0", "D_M", len(mon), len(mon))
        for nm in RC.TH:
            put("G_ck_theta", k, nm + "|median", "months θ≠0", len(mon), mon[nm].median())
            put("G_ck_theta", k, nm + "|share_ne0", "months θ≠0", len(mon), (mon[nm].abs() > 1e-6).mean())
        dl = np.log(S[k]["Cc"].to_numpy(float) / c8)
        put("G_ck_theta", k, "share_days_lnM_lt0", "D_M", len(x), (dl < -1e-12).mean())


def sec_level_score(x, a, S):
    """Балл уровня в θ = 0 по пулу месяца (закрытые пары t⁺ ≤ D): ∂L/∂(θ_c0 + θ_p0). Уровень F (R): mean π(1 − min(x,c)/m); уровень C: mean(1 − min(x_C,c_C)/m_C) ≡ 0.
    Положительный балл ⇒ обучение уводит уровень C вниз. Описательно, по сохранённым π (PIT)."""
    rv, d, tp, pi = x["rv"].to_numpy(float), pd.DatetimeIndex(x["d"]), pd.DatetimeIndex(x["t_plus"]), x["pi"].to_numpy(float)
    w5 = pd.read_parquet(C.ROOT / "data/regen/harj_x1/W5/w5g.parquet")
    xF, xC = rv / w5["F"].to_numpy(float), rv / a["c8"]
    rck = S["S-RCK"].assign(mo=pd.to_datetime(S["S-RCK"]["d"]).dt.to_period("M")).drop_duplicates("mo")
    out = []
    for _, r in rck[rck["active"]].iterrows():
        P = np.where(tp <= pd.Timestamp(r["d"]))[0]
        cF = np.quantile(xF[P], 0.99)
        mF = np.minimum(xF[P], cF).mean()
        cC = np.quantile(xC[P], 0.99)
        mC = np.minimum(xC[P], cC).mean()
        yF, yC = np.minimum(xF[P], cF) / mF, np.minimum(xC[P], cC) / mC
        out.append(dict(lam=r["lam"], sF=float(np.mean(pi[P] * (1 - yF))), sC=float(np.mean(1 - yC)), cov=float(np.cov(pi[P], yF)[0, 1]),
                        sF_plain=float(np.mean(pi[P] * (1 - xF[P]))), pibar=float(pi[P].mean()), mF=mF))
    o = pd.DataFrame(out)
    n = len(o)
    put("H_level_score", "S-RCK", "score_levelF_R|median", "active months", n, o["sF"].median())
    put("H_level_score", "S-RCK", "score_levelF_R|share_pos", "active months", n, (o["sF"] > 0).mean())
    put("H_level_score", "S-RCK", "score_levelC_R|max_abs", "active months", n, o["sC"].abs().max())
    put("H_level_score", "S-RCK", "cov_pi_clipped_ratio|median", "active months", n, o["cov"].median())
    put("H_level_score", "S-RCK", "score_levelF_plain|median", "active months", n, o["sF_plain"].median())
    put("H_level_score", "S-RCK", "score_levelF_plain|share_pos", "active months", n, (o["sF_plain"] > 0).mean())
    put("H_level_score", "S-RCK", "pi_mean_pool|median", "active months", n, o["pibar"].median())
    put("H_level_score", "S-RCK", "one_minus_m|median", "active months", n, 1 - o["mF"].median())
    # описательно, не PIT: средний ln(RV/W5-G) по терцилям π (вся D_M)
    q = np.quantile(pi, [1 / 3, 2 / 3])
    xB = rv / x["M"].to_numpy(float)
    for i, (lo, hi) in enumerate(((-1, q[0]), (q[0], q[1]), (q[1], 2))):
        m = (pi > lo) & (pi <= hi)
        for nm, xx in (("W5-G", xF), ("cfg8", xC), ("B_mix", xB)):
            put("H_level_score", nm, "mean_clipped_ratio_pi_tercile%d" % (i + 1), "D_M (описательно, не PIT)", m.sum(), np.minimum(xx, np.quantile(xx, 0.99))[m].mean())
        put("H_level_score", "S-RCK", "mean_lnC_cfg8_pi_tercile%d" % (i + 1), "D_M (описательно, не PIT)", m.sum(), np.log(S["S-RCK"]["Cc"].to_numpy(float) / a["c8"])[m].mean())
    yr = rck[rck["active"] & (rck["lam"] > 0)].assign(yr=pd.to_datetime(rck["d"]).dt.year)
    for y, g in yr.groupby("yr"):
        put("H_level_score", "S-RCK", "th_c0_plus_th_p0_half|mean", str(y), len(g), (0.5 * (g["th_c0"] + g["th_p0"])).mean())


def b_dev(c4):
    E = np.exp(AF.spline_basis(AF.TAU) @ np.asarray(c4, float))
    b = E / E.mean()
    return float(np.max(np.abs(b - 1.0)))


def a_tv(k, om, psi):
    L_, R_ = AF.LAGS[k], AF.RECT[k]
    ok = np.ones((1, L_), bool)
    a = AF.lag_weights(ok, k, torch.tensor(om, dtype=torch.float64), torch.tensor(psi, dtype=torch.float64)).numpy()[0]
    r = (np.arange(L_) < R_) / R_
    return 0.5 * float(np.abs(a - r).sum())


def sec_a2_params(S):
    """Параметры A2 по месяцам: старый S-A2 против S-RA2/b/c (θ — среднее по seeds; b и a — по среднему θ, приближённо)."""
    for k in ("S-A2", "S-RA2", "S-RA2b", "S-RA2c"):
        mon = S[k].assign(mo=pd.to_datetime(S[k]["d"]).dt.to_period("M"), yr=pd.to_datetime(S[k]["d"]).dt.year).drop_duplicates("mo")
        nz = mon[[c for c in mon.columns if c.startswith(("d_", "om_", "c_", "psi_")) or c in ("d0", "dw", "dk")]].abs().sum(axis=1) > 0
        mon = mon[nz]
        for i, kk in enumerate("sml"):
            mon["bdev_" + kk] = [b_dev(r[["c_%s%d" % (kk, j) for j in (1, 2, 3, 4)]].to_numpy(float)) for _, r in mon.iterrows()]
            mon["atv_" + kk] = [a_tv(i, r["om_" + kk], r[["psi_%s1" % kk, "psi_%s2" % kk]].to_numpy(float)) for _, r in mon.iterrows()]
        cols = ["lam", "d0", "d_s", "d_m", "d_l", "d_J", "om_s", "om_m", "om_l", "dw", "dk", "bdev_s", "bdev_m", "bdev_l", "atv_s", "atv_m", "atv_l"]
        put("I_a2_params", k, "months_theta_ne0", "D_M", len(mon), len(mon))
        for c in cols:
            put("I_a2_params", k, c + "|median", "months θ≠0", len(mon), mon[c].median())
        for y, g in mon.groupby("yr"):
            for c in ("lam", "d0", "om_s", "bdev_s", "bdev_l", "dw", "dk"):
                put("I_a2_params_year", k, c + "|mean", str(y), len(g), g[c].mean())


def sec_layer(x, gap, w5, bg, S):
    """Слой доверия над новым C: среднее π против X1 на днях C ≠ cfg8; связь δ = ln(C/cfg8) с уровнем ln cfg8; уровень против разброса на уровне F при замороженном π."""
    rv, c8, lw = x["rv"].to_numpy(float), x["C"].to_numpy(float), ell(x["rv"], w5["F"])
    for q in (0.05, 0.5, 0.95):
        put("J_layer", "cfg8", "q%02d_ln_cfg8" % int(q * 100), "D_M", len(x), np.quantile(np.log(c8), q))
    for k in ("S-CK", "S-RCK", "S-RCKb", "S-RCKc", "S-A2", "S-RA2", "S-RA2b", "S-RA2c"):
        df = S[k]
        dl = np.log(df["Cc"].to_numpy(float) / c8)
        act = np.abs(dl) > 1e-12
        pi_new = L.layers(x, df["Cc"].to_numpy(float), gap)["pi"].to_numpy(float)
        put("J_layer", k, "pi_mean_new", "C ≠ cfg8", act.sum(), pi_new[act].mean())
        put("J_layer", k, "pi_mean_X1", "C ≠ cfg8", act.sum(), x["pi"].to_numpy(float)[act].mean())
        put("J_layer", k, "corr_delta_ln_cfg8", "C ≠ cfg8", act.sum(), np.corrcoef(dl[act], np.log(c8[act]))[0, 1])
        mu = dl[act].mean()
        lvl = np.where(act, c8 * np.exp(mu), c8)
        disp = np.where(act, c8 * np.exp(dl - mu), c8)
        put("J_layer", k, "Ffr_only_level_vs_W5G", "D_M", len(x), dpct(ell(rv, ffr(x, bg, lvl)), lw), "W5-G")
        put("J_layer", k, "Ffr_only_dispersion_vs_W5G", "D_M", len(x), dpct(ell(rv, ffr(x, bg, disp)), lw), "W5-G")


def main():
    x, gap, a, w5, bg, S = load()
    sec_layer(x, gap, w5, bg, S)
    sec_table(x, w5, bg, S)
    sec_cdelta(x, S)
    sec_months(x, w5, bg, S)
    sec_ck_parts(x, gap, a, w5, bg, S)
    sec_level_score(x, a, S)
    sec_a2_params(S)
    df = pd.DataFrame(ROWS)
    head = ["# S-D (Ж-256): разбор S-R по сохранённым рядам; DEVELOPMENT / HISTORICAL WALK-FORWARD; исследовательский, не OOS; обучения нет (слои X1/W5 пересчитаны прежним кодом над C при фиксированных θ)",
            "# D_M: t⁺ 2018-01-04…2026-11-10, n = 2198; l = RV/F − ln(RV/F) − 1; Δ, % — 100·(QLIKE ряда − QLIKE базы)/QLIKE базы на окне строки, минус — лучше; contrib — п. п. QLIKE базы на D_M (Σ по разбиению = Δ)",
            "# Ffr = B_mix^(1−π)·C^π·e^{βg} с сохранёнными π (X1) и β·g (W5-G); генератор <sp>/s_d/expl.py"]
    OUT.write_text("\n".join(head) + "\n" + df.to_csv(index=False, float_format="%.6g"), encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_rows", 2000, "display.max_colwidth", 60)
    print(df.to_string(index=False, float_format=lambda v: "%.6g" % v))


if __name__ == "__main__":
    main()
