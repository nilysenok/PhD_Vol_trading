"""W3 D-π: таблица `docs/otchety/harj_x1/W3_DIAG.csv` и проверки. Запуск из корня репозитория.
python3 -m models_v2.s16.x1_w3_diag --write — счёт и запись CSV (единственный пишущий режим; перенос из scratchpad, W4-A: перезапись только явным флагом, days.parquet не пишется)
python3 -m models_v2.s16.x1_w3_diag --check — пересчёт и побитная сверка строк данных (без шапки-комментария) с файлом; ничего не пишет (sha файла до и после)
--check-join | --check-repro | --check-ident | --check-pit | --check-known | --check-verdict — только читают."""
from __future__ import annotations

import hashlib
import itertools
import sys

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w3_core as C

OUT = C.ROOT / "docs/otchety/harj_x1/W3_DIAG.csv"
PERIODS = [str(y) for y in range(2018, 2026)] + ["2026a", "2026F"]
WINDOWS = {"M": PERIODS, "2018_2021": PERIODS[:4], "2022_2026": PERIODS[4:], "2019_2026_posthoc": PERIODS[1:]}  # последнее — после просмотра, вне вердикта
SCHEMES = C.MAIN_SCHEMES + C.SENS_SCHEMES
W3 = [w for w in itertools.product(np.round(np.arange(21) * 0.05, 2), repeat=2) if w[0] + w[1] <= 1.0 + 1e-9]


def build():
    x = C.load()
    rv, M, Cc = x["rv"].to_numpy(), x["M"].to_numpy(), x["C"].to_numpy()
    L = C.grid_losses(rv, M, Cc)
    los = {k: C.qlike(rv, x[k]) for k in ("F", "M", "C", "K1", "K2", "K3")}
    los["X1"] = los.pop("F")
    s = x["s"].to_numpy(int)
    alpha = np.where(s == 1, x["a1"], x["a0"])
    _, eta, _ = C.fs_pass(los["C"], los["M"], s, x["d"].to_numpy(), x["t_plus"].to_numpy(), x[["a0", "a1"]].to_numpy())
    pis = {}
    for nm in SCHEMES:
        pis[nm] = C.scheme(nm, x, L)
        los[nm] = C.qlike(rv, C.combine(M, Cc, pis[nm]))
    # три эксперта ex post: веса (w_M, w_C, w_K1 = 1 − w_M − w_C)
    lnM, lnC, lnK = np.log(M), np.log(Cc), np.log(x["K1"].to_numpy())
    L3 = np.column_stack([C.qlike(rv, np.exp(a * lnM + b * lnC + (1 - a - b) * lnK)) for a, b in W3])
    # π = «доля лет»: доля прошлых полных лет t⁺ (≥ 2018), в которых QLIKE(C) < QLIKE(M) в своём состоянии; 2018 — 0,5 (PIT по годам)
    yr = x["t_plus"].dt.year.to_numpy()
    better = {(y, k): float(los["C"][(yr == y) & (s == k)].mean() < los["M"][(yr == y) & (s == k)].mean()) for y in range(2018, 2026) for k in (0, 1)}
    piy = np.array([0.5 if y == 2018 else np.mean([better[(yy, k)] for yy in range(2018, y)]) for y, k in zip(yr, s)])
    los["PIYEARS"] = C.qlike(rv, C.combine(M, Cc, piy))
    days = x[["d", "t_plus", "rv", "M", "C", "K1", "K2", "K3", "F", "s", "pi", "per"]].copy()
    days["eta"], days["alpha"], days["pi_years"] = eta, alpha, piy
    for nm in SCHEMES:
        days["pi_" + nm] = pis[nm]
    for k, v in los.items():
        days["l_" + k] = v
    return x, L, L3, los, days


def cell_rows(x, L, L3, los, days):
    s, per = x["s"].to_numpy(int), x["per"].to_numpy()
    nM, qK2M = len(x), los["K2"].mean()
    D = los["C"] - los["M"]
    rows, best = [], {}
    for p in PERIODS:
        for st in ("N", "E", "all"):
            m = (per == p) & ((s == 0) if st == "N" else (s == 1) if st == "E" else True)
            n = int(m.sum())
            if n == 0:
                rows.append(dict(block="cell", period=p, state=st, n=0))
                continue
            r = dict(block="cell", period=p, state=st, n=n, pi_mean_x1=x["pi"][m].mean(), pi_std_x1=float(x["pi"][m].std(ddof=1)) if n > 1 else np.nan, alpha_mean=days["alpha"][m].mean(), eta_med=float(np.nanmedian(days["eta"][m])),  # η дня 1 — NaN (пар ещё нет)
                     q_M=los["M"][m].mean(), q_C=los["C"][m].mean(), dqC_vs_M_pct=(los["C"][m].mean() / los["M"][m].mean() - 1) * 100,
                     q_X1=los["X1"][m].mean(), q_K1=los["K1"][m].mean(), q_K2=los["K2"][m].mean(), q_K3=los["K3"][m].mean(),
                     dX1_vs_K2_pct=(los["X1"][m].mean() / los["K2"][m].mean() - 1) * 100, dX1_vs_M_pct=(los["X1"][m].mean() / los["M"][m].mean() - 1) * 100,
                     contrib_X1_K2_pp=(los["X1"][m] - los["K2"][m]).sum() / (nM * qK2M) * 100,
                     contrib_X1_M_pp=(los["X1"][m] - los["M"][m]).sum() / (nM * qK2M) * 100,
                     contrib_M_K2_pp=(los["M"][m] - los["K2"][m]).sum() / (nM * qK2M) * 100,
                     D_mean=D[m].mean(), D_median=float(np.median(D[m])), frac_C_worse=float((D[m] > 0).mean()),
                     top5pct_share_absD=float(np.sort(np.abs(D[m]))[::-1][: max(1, int(np.ceil(0.05 * n)))].sum() / np.abs(D[m]).sum()),
                     q_pi_years=los["PIYEARS"][m].mean(), pi_years=days["pi_years"][m].mean(),
                     corr_lnCM_lnK1M=float(np.corrcoef(np.log(x["C"][m] / x["M"][m]), np.log(x["K1"][m] / x["M"][m]))[0, 1]) if n > 3 else np.nan)
            if st != "all":
                g = int(np.argmin(L[m].mean(0)))
                best[(p, st)] = g
                g3 = int(np.argmin(L3[m].mean(0)))
                r.update(pi_ceil=C.GRID[g], q_ceil=L[m, g].mean(), q_pi_mean=C.qlike(x["rv"][m], C.combine(x["M"][m], x["C"][m], np.full(n, x["pi"][m].mean()))).mean(),
                         w3_ceil="M%.2f/C%.2f/K1%.2f" % (W3[g3][0], W3[g3][1], 1 - W3[g3][0] - W3[g3][1]), q_ceil3=L3[m, g3].mean(),
                         fs_pi_star=C.fs_fixed_point(D[m].mean(), r["eta_med"], r["alpha_mean"]), fs_pi_252=C.fs_after(D[m].mean(), r["eta_med"], r["alpha_mean"]),
                         fs_pi_floor=r["alpha_mean"] / 2)
                r["timing_X1_pct"] = (r["q_X1"] / r["q_pi_mean"] - 1) * 100
                r["gap_ceil_pp"] = (los["X1"][m] - L[m, g]).sum() / (nM * los["X1"].mean()) * 100
                sc = float(np.mean(x["rv"][m] / x["F"][m]))
                r["level_ceil_scale"], r["q_level_ceil"] = sc, C.qlike(x["rv"][m], x["F"][m] * sc).mean()
            else:
                gp = int(np.argmin(L[m].mean(0)))
                r.update(pi_ceil_pooled=C.GRID[gp], q_ceil_pooled=L[m, gp].mean())
            for nm in SCHEMES:
                r["q_" + nm], r["pi_mean_" + nm] = los[nm][m].mean(), days["pi_" + nm][m].mean()
            rows.append(r)
    t = pd.DataFrame(rows)
    # клетка all: потолок = сумма потолков N и E; π константа по состояниям
    for p in PERIODS:
        ia = t.index[(t.period == p) & (t.state == "all")][0]
        sub = t[(t.period == p) & (t.state != "all") & (t.n > 0)]
        t.loc[ia, "q_ceil"] = (sub["q_ceil"] * sub["n"]).sum() / sub["n"].sum()
        t.loc[ia, "q_ceil3"] = (sub["q_ceil3"] * sub["n"]).sum() / sub["n"].sum()
        t.loc[ia, "q_pi_mean"] = (sub["q_pi_mean"] * sub["n"]).sum() / sub["n"].sum()
        t.loc[ia, "q_level_ceil"] = (sub["q_level_ceil"] * sub["n"]).sum() / sub["n"].sum()
        t.loc[ia, "gap_ceil_pp"] = sub["gap_ceil_pp"].sum()
        t.loc[ia, "timing_X1_pct"] = (t.loc[ia, "q_X1"] / t.loc[ia, "q_pi_mean"] - 1) * 100
        t.loc[ia, "pi_ceil"] = np.nan
    return t, best


def window_rows(x, los, t):
    per = x["per"].to_numpy()
    s = x["s"].to_numpy(int)
    rows = []
    for w, ps in WINDOWS.items():
        for st in ("all", "N", "E"):
            m = np.isin(per, ps) & ((s == 0) if st == "N" else (s == 1) if st == "E" else True)
            n = int(m.sum())
            c = t[t.period.isin(ps) & (t.state == st) & (t.n > 0)]
            r = dict(block="window", period=w, state=st, n=n, pi_mean_x1=x["pi"][m].mean(), q_M=los["M"][m].mean(), q_C=los["C"][m].mean(),
                     dqC_vs_M_pct=(los["C"][m].mean() / los["M"][m].mean() - 1) * 100, q_X1=los["X1"][m].mean(), q_K1=los["K1"][m].mean(),
                     q_K2=los["K2"][m].mean(), q_K3=los["K3"][m].mean(), dX1_vs_K2_pct=(los["X1"][m].mean() / los["K2"][m].mean() - 1) * 100,
                     dX1_vs_M_pct=(los["X1"][m].mean() / los["M"][m].mean() - 1) * 100,
                     q_ceil=(c["q_ceil"] * c["n"]).sum() / n, q_ceil3=(c["q_ceil3"] * c["n"]).sum() / n, q_pi_mean=(c["q_pi_mean"] * c["n"]).sum() / n,
                     q_pi_years=los["PIYEARS"][m].mean(), q_level_ceil=(c["q_level_ceil"] * c["n"]).sum() / n,
                     timing_X1_pct=(los["X1"][m].mean() / ((c["q_pi_mean"] * c["n"]).sum() / n) - 1) * 100)
            r.update(persistence(x, los, m))
            if st == "all":
                r["q_ceil_pooled"] = (c["q_ceil_pooled"] * c["n"]).sum() / n
            gap = r["q_X1"] - r["q_ceil"]
            for nm in SCHEMES:
                r["q_" + nm] = los[nm][m].mean()
                r["capture_" + nm] = (r["q_X1"] - r["q_" + nm]) / gap if gap > 0 else np.nan
                r["d_%s_vs_X1_pct" % nm] = (r["q_" + nm] / r["q_X1"] - 1) * 100
            r["capture_M"] = (r["q_X1"] - r["q_M"]) / gap if gap > 0 else np.nan
            r["capture_PIYEARS"] = (r["q_X1"] - r["q_pi_years"]) / gap if gap > 0 else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def persistence(x, los, m):
    """Помесячная устойчивость (календарный месяц t⁺, ≥ 5 пар в клетке): D = QLIKE(C) − QLIKE(M), a = ln mean(RV/X1).
    D_month_ac1 / a_month_ac1 — корреляция месяца с предыдущим; D_month_sign_persist — доля месяцев с тем же знаком D, что у предыдущего;
    corr_sig2prev_a — корреляция дисперсии ln(RV/X1) прошлого месяца с a текущего (сигнал «vol-of-vol → уровень», Б5)."""
    df = pd.DataFrame({"mon": x["t_plus"].dt.to_period("M").to_numpy()[m], "D": (los["C"] - los["M"])[m],
                       "ratio": (x["rv"] / x["F"]).to_numpy()[m], "u": np.log(x["rv"] / x["F"]).to_numpy()[m]})
    g = df.groupby("mon").agg(n=("D", "size"), D=("D", "mean"), r=("ratio", "mean"), v=("u", "var"))
    g = g[g.n >= 5]
    g["a"] = np.log(g["r"])
    cons = g.index.to_series().diff().apply(lambda z: z.n if pd.notna(z) else np.nan).to_numpy() == 1
    D0, D1 = g["D"].to_numpy()[:-1][cons[1:]], g["D"].to_numpy()[1:][cons[1:]]
    a0, a1, v0 = g["a"].to_numpy()[:-1][cons[1:]], g["a"].to_numpy()[1:][cons[1:]], g["v"].to_numpy()[:-1][cons[1:]]
    cc = lambda p, q: float(np.corrcoef(p, q)[0, 1]) if len(p) > 3 else np.nan  # noqa: E731
    return dict(n_month_pairs=int(len(D0)), D_month_ac1=cc(D0, D1), D_month_sign_persist=float(np.mean(np.sign(D0) == np.sign(D1))) if len(D0) else np.nan,
                a_month_ac1=cc(a0, a1), corr_sig2prev_a=cc(v0, a1))


def posthoc_rows(days):
    """ПОСЛЕ ПРОСМОТРА (вне вердикта), block = posthoc: π в N заменён (N05: 0,5; N_AH; N_DFTL126), в E — π X1; E_M: в E π = 0, в N — X1."""
    N, lx = (days.s == 0).to_numpy(), days.l_X1.to_numpy()
    var = {"N05": np.where(N, 0.5, days.pi), "N_AH": np.where(N, days.pi_AH, days.pi), "N_DFTL126": np.where(N, days.pi_DFTL126, days.pi), "E_M": np.where(N, days.pi, 0.0)}
    per, rows = days.per.to_numpy(), []
    for nm, pi in var.items():
        l = C.qlike(days.rv, C.combine(days.M, days.C, pi))
        for p, ps in list(WINDOWS.items())[:3] + [(p, [p]) for p in PERIODS]:
            m = np.isin(per, ps)
            rows.append(dict(block="posthoc", variant=nm, period=p, state="all", n=int(m.sum()), q_variant=l[m].mean(), q_X1=lx[m].mean(),
                             d_variant_vs_X1_pct=(l[m].mean() / lx[m].mean() - 1) * 100))
    return pd.DataFrame(rows)


HEADER = ("# DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; диагностика после просмотра окна M (Ж-250 п. 3–4, лист W3 D-π); "
          "потолки q_ceil*, q_pi_mean, pi_ceil* ИСПОЛЬЗУЮТ ИСХОД (не прогноз); PIT-схемы — диагностика потолка механизма, не отбор; "
          "l = RV/F − ln(RV/F) − 1, среднее дневных потерь; вход sha256: %s; код w3/core.py+diag.py (scratchpad архитектора)")


def data_text():
    """Строки данных (имена колонок + значения) как в файле, без шапки-комментария."""
    x, L, L3, los, days = build()
    t, _ = cell_rows(x, L, L3, los, days)
    w = window_rows(x, los, t)
    out = pd.concat([t, w, posthoc_rows(days)], ignore_index=True)
    return out.to_csv(index=False, float_format="%.6g")


def write():
    head = HEADER % "; ".join("%s=%s" % (k, C.sha(v)[:16]) for k, v in C.SRC.items())
    text = data_text()
    OUT.write_text(head + "\n" + text, encoding="utf-8")
    print("wrote", OUT, text.count("\n") - 1, "rows")


def check():
    """`--check`: строки данных файла (после первой строки-шапки) == пересчёт побитно; файл не изменён."""
    h0 = hashlib.sha256(OUT.read_bytes()).hexdigest()
    saved = OUT.read_text(encoding="utf-8").split("\n", 1)[1]
    new = data_text()
    h1 = hashlib.sha256(OUT.read_bytes()).hexdigest()
    ok = saved == new and h0 == h1
    print("check %s: строк данных файла %d, пересчёта %d; совпадение побитно %s; sha до = после %s" % ("ok" if ok else "FAIL", saved.count("\n") - 1, new.count("\n") - 1, saved == new, h0 == h1))
    return ok


def check_join():
    x = C.load()
    print("join ok n=%d %s %s; E-days %d; periods %s" % (len(x), x.t_plus.min().date(), x.t_plus.max().date(), int(x.s.sum()), dict(pd.Series(x.per).value_counts().sort_index())))


def check_repro():
    x = C.load()
    qc, qm = C.qlike(x.rv, x.C), C.qlike(x.rv, x.M)
    pi, _, pis = C.fs_pass(qc, qm, x.s.to_numpy(int), x.d.to_numpy(), x.t_plus.to_numpy(), x[["a0", "a1"]].to_numpy())
    e1 = float(np.max(np.abs(pi - x.pi)))
    e1b = float(np.max(np.abs(pis - x[["pi0", "pi1"]].to_numpy())))
    eF = float(np.max(np.abs(C.combine(x.M, x.C, x.pi) - x.F) / x.F))
    sys.path.append(str(C.ROOT))
    import models_v2.s15  # noqa: F401  (torch первым)
    from models_v2.s15 import de_e1_state as E
    r = E.run(x.rv, x.M, x.C, x.s.to_numpy(int), x.d.to_numpy(), x.t_plus.to_numpy())
    e2 = float(np.max(np.abs(r["pi"] - x.pi)))
    ea = float(max(np.max(np.abs(r["a0"] - x.a0)), np.max(np.abs(r["a1"] - x.a1))))
    ok = max(e1, e1b, eF, e2, ea) < 1e-12
    print("repro %s: own pass vs saved pi %.2e (pi0/pi1 %.2e); F=combine %.2e; de_e1_state.run vs saved pi %.2e, alpha %.2e" % ("ok" if ok else "FAIL", e1, e1b, eF, e2, ea))
    return ok


def check_ident():
    x = C.load()
    L = C.grid_losses(x.rv.to_numpy(), x.M.to_numpy(), x.C.to_numpy())
    i0 = np.array_equal(L[:, 0], C.qlike(x.rv, x.M)) and np.array_equal(L[:, -1], C.qlike(x.rv, x.C))
    t = pd.read_csv(OUT, comment="#")
    c = t[(t.block == "cell") & (t.state != "all") & (t.n > 0)]
    i1 = bool((c.q_ceil <= np.minimum(c.q_M, c.q_C) + 1e-12).all())
    beat = c[c.q_X1 < c.q_ceil][["period", "state", "q_X1", "q_ceil"]]
    print("ident %s: grid ends = M, C bitwise %s; ceiling ≤ min(M, C) in all cells %s; cells where X1 < const-ceiling: %d %s"
          % ("ok" if i0 and i1 else "FAIL", i0, i1, len(beat), beat.to_dict("records")))


def check_pit():
    x = C.load()
    rv = x.rv.to_numpy().copy()
    L0 = C.grid_losses(rv, x.M.to_numpy(), x.C.to_numpy())
    k = 1200
    bad = x.t_plus.to_numpy() > x.d.to_numpy()[k]
    rv2 = rv.copy()
    rv2[bad] *= 7.0
    L1 = C.grid_losses(rv2, x.M.to_numpy(), x.C.to_numpy())
    x2 = x.copy()
    x2["rv"] = rv2
    res = {}
    for nm in ("DFTL126", "AH", "HEDGE0"):
        a, b = C.scheme(nm, x, L0), C.scheme(nm, x2, L1)
        res[nm] = bool(np.array_equal(a[: k + 1], b[: k + 1]))
    orig = C.closed_date
    C.closed_date = lambda tp, d: d  # мутация «закрыто + 1»
    try:
        mut = {nm: not np.array_equal(C.scheme(nm, x, L0)[: k + 1], C.scheme(nm, x2, L1)[: k + 1]) for nm in ("DFTL126", "AH", "HEDGE0")}
    finally:
        C.closed_date = orig
    ok = all(res.values()) and all(mut.values())
    print("pit %s: honest unchanged %s; mutation caught %s" % ("ok" if ok else "FAIL", res, mut))


def check_known():
    """Известный ответ: потери минимальны при π = 0,3 каждый день (квадратичные по π) -> DFTL ровно 0,3; AH сходится к 0,3 (± подмешивание α = 0,01 к равным весам)."""
    n = 400
    rng = np.random.default_rng(0)
    L = (C.GRID[None, :] - 0.3) ** 2 + 0.01 * rng.standard_normal((n, 1))
    d = np.arange(n).astype("datetime64[D]")
    tp = d + 1
    s = np.zeros(n, int)
    a, b = C.dftl(L, s, d, tp, 126.0), C.adahedge_fs(L, s, d, tp, 0.01)
    ok = bool(np.all(a[5:] == 0.3)) and abs(b[-50:].mean() - 0.3) < 0.01
    print("known %s: DFTL tail %s; AH tail mean %.4f (ожидание 0,3 + 0,01·(0,5 − 0,3) ≈ 0,302)" % ("ok" if ok else "FAIL", np.unique(a[5:]), b[-50:].mean()))


def check_verdict():
    t = pd.read_csv(OUT, comment="#")
    w = t[(t.block == "window") & (t.period == "M") & (t.state == "all")].iloc[0]
    cap = {nm: w["capture_" + nm] for nm in C.MAIN_SCHEMES}
    v = max(cap.values()) >= 1 / 3
    print("verdict: gap X1→ceiling %.6f (X1/ceiling − 1 = %.2f %%); capture %s; sensitivity %s; M %.3f; PIYEARS %.3f → %s"
          % (w.q_X1 - w.q_ceil, (w.q_X1 / w.q_ceil - 1) * 100, {k: round(v_, 3) for k, v_ in cap.items()},
             {nm: round(w["capture_" + nm], 3) for nm in C.SENS_SCHEMES}, w.capture_M, w.capture_PIYEARS,
             "ПОДТВЕРЖДЕНА (≥ 1/3)" if v else "НЕ ПОДТВЕРЖДЕНА (< 1/3)"))


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "--check"
    r = {"--write": write, "--check": check, "--check-join": check_join, "--check-repro": check_repro, "--check-ident": check_ident, "--check-pit": check_pit, "--check-verdict": check_verdict, "--check-known": check_known}[a]()
    sys.exit(0 if r is not False else 1)
