"""FWD2: мощность DM по сохранённым рядам окна M (t+ 2018-01-04…2026-11-10, n = 2198). Только чтение; новых данных нет.
Объявление — DECL.md (записан до запуска). Выход — POWER2.csv рядом со скриптом. Запуск: python3 power2.py [--check]."""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

ROOT = Path("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading")
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

from models_v2.s3 import dm as dmod  # noqa: E402
from models_v2.s16 import x1_w3_core as C  # noqa: E402
from models_v2.s16 import x1_w8_gap_z as ZG  # noqa: E402

OUT = Path(__file__).resolve().parent / "POWER2.csv"
ALPHA, M_FAM, LAG, WIN = 0.05, 7, 5, 21
EFFECTS = (-0.3, -0.5, -1.0)
POWERS = (0.5, 0.8)
HORIZONS = (250, 500, 750, 1000, 1233, 1250, 2500)
SRC = {"w5g": "data/regen/harj_x1/W5/w5g.parquet", "w8g": "data/regen/harj_x1/W8/w8g.parquet"}


def sha16(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()[:16]


def load():
    x = C.load()
    w5, w8 = (pd.read_parquet(ROOT / SRC[k]) for k in ("w5g", "w8g"))
    for w in (w5, w8):
        assert len(w) == len(x) and (pd.to_datetime(w["d"]).to_numpy() == x["d"].to_numpy()).all()
        assert np.array_equal(w["X1"].to_numpy(), x["F"].to_numpy()) and np.array_equal(w["rv"].to_numpy(), x["rv"].to_numpy())
    e = x["s"].to_numpy(int)
    assert set(np.unique(e)) <= {0, 1}
    sh = pd.Series(e).rolling(WIN, min_periods=1).max().to_numpy(int)       # PIT: строки j ≤ d (21 строка календаря пар)
    rv = x["rv"].to_numpy(float)
    L = {"X1": C.qlike(rv, x["F"]), "K1": C.qlike(rv, x["K1"]), "K2": C.qlike(rv, x["K2"]), "K3": C.qlike(rv, x["K3"]),
         "W5G": C.qlike(rv, w5["F"]), "W8G": C.qlike(rv, w8["F"])}
    gap = w5["gap"].to_numpy(int)
    on = gap.astype(bool) & w5["refit_date"].notna().to_numpy()
    return x, L, gap, on, sh, shock_month_v2(x["d"])


def shock_month_v2(d):
    """DECL v2: G(j) — PIT-терциль ln RV^IMOEX (замороженный код W8); месяц d — шок, если доля T3 среди G ≠ na > 1/2. -1 — не определено."""
    g = ZG.labels(ZG.z_series(d))
    df = pd.DataFrame({"m": pd.DatetimeIndex(d).to_period("M"), "t3": g == "T3", "ok": g != "na"})
    agg = df.groupby("m").agg(t3=("t3", "sum"), ok=("ok", "sum"))
    lab = np.where(agg["ok"] == 0, -1, (agg["t3"] / agg["ok"].where(agg["ok"] > 0, 1) > 0.5).astype(int))
    return pd.Series(lab, index=agg.index).reindex(df["m"]).to_numpy(int)


def stats(D, LB):
    """D = l_A − l_B; σ_LR (НУ, лаг 5, = n·Var(среднего)); μ, Q_B, Δ %; DM-стат одностор. «A лучше» (мин. = D<0)."""
    n = len(D)
    s2 = dmod.nw_variance(D, LAG) * n
    mu, qb = float(D.mean()), float(LB.mean())
    return dict(n=n, sigma=math.sqrt(s2), mu=mu, qb=qb, delta_pct=100 * mu / qb, dm=mu / math.sqrt(s2 / n))


def n_req(sigma, qb, delta_pct, alpha, power):
    mu = abs(delta_pct) / 100 * qb
    return ((norm.ppf(1 - alpha) + norm.ppf(power)) * sigma / mu) ** 2


def pw(sigma, qb, delta_pct, alpha, n):
    mu = abs(delta_pct) / 100 * qb
    return float(norm.cdf(mu * math.sqrt(n) / sigma - norm.ppf(1 - alpha)))


def boot_power(D, LB, delta_pct, n, alpha, sign, R=400, block=10, seed=0):
    """Кольцевой блочный бутстрап центрированных D; сдвиг на днях D ≠ 0 (суммарно δ·Q_B на день); доля отвержений DM."""
    rng = np.random.default_rng(seed)
    nz = D != 0
    mu = sign * abs(delta_pct) / 100 * LB.mean()
    Dc = D - D.mean()
    Ds = Dc + np.where(nz, mu * len(D) / nz.sum(), 0.0)
    n0, nb = len(D), -(-n // block)
    z = norm.ppf(1 - alpha)
    rej = 0
    for _ in range(R):
        st = rng.integers(0, n0, nb)
        idx = ((st[:, None] + np.arange(block)[None, :]) % n0).ravel()[:n]
        y = Ds[idx]
        t = y.mean() / math.sqrt(dmod.nw_variance(y, LAG))
        rej += (t < -z) if sign < 0 else (t > z)
    return rej / R


def main():
    x, L, gap, on, sh, sm = load()
    yr = x["t_plus"].dt.year.to_numpy()
    fam = {"H1": ("W5G", "X1"), "H2": ("W8G", "W5G"), "H3": ("W5G", "K2"), "H4": ("W5G", "K3"), "H5": ("X1", "K2"), "H6": ("X1", "K3")}
    masks = {"M": np.ones(len(x), bool), "M_bez_2022": yr != 2022, "2023_2026": yr >= 2023}
    rows = []
    for h, (a, b) in fam.items():
        for w, m in masks.items():
            s = stats(L[a][m] - L[b][m], L[b][m])
            base = dict(hyp=h, A=a, B=b, noise=w, n_hist=s["n"], Q_B=s["qb"], sigma_LR=s["sigma"], delta_hist_pct=s["delta_pct"], dm_hist=s["dm"],
                        share_nonzero=float((L[a][m] - L[b][m] != 0).mean()))
            for dl in EFFECTS + (s["delta_pct"],):
                tag = "hist" if dl == s["delta_pct"] else "%+.1f" % dl
                r = dict(base, effect=tag, effect_pct=dl)
                if dl >= 0:
                    rows.append(dict(r, note="исторический эффект ≥ 0: мощность не определена"))
                    continue
                for al, an in ((ALPHA, "a05"), (ALPHA / M_FAM, "holm7")):
                    for p in POWERS:
                        r["n_p%02d_%s" % (int(p * 100), an)] = n_req(s["sigma"], s["qb"], dl, al, p)
                    for hz in HORIZONS:
                        r["pow_n%d_%s" % (hz, an)] = pw(s["sigma"], s["qb"], dl, al, hz)
                rows.append(r)
    # H7 — механизм: C = gap·on·(спокойно), премия: l_W5G − l_X1 > 0. v1 — метка e за 21 строку (DECL п. 2, негодна); v2 — месяц IMOEX (DECL ↺ v2)
    calm = {"H7v1": sh == 0, "H7": sm == 0}
    shock = {"H7v1": sh == 1, "H7": sm == 1}
    for ver in ("H7v1", "H7"):
        for w, m in masks.items():
            for part, sel in (("C_calm", on & calm[ver]), ("S_shock", on & shock[ver])):
                mm = m & sel
                D, LB = L["W5G"][mm] - L["X1"][mm], L["X1"][mm]
                s = stats(D, LB)
                share = float(mm.sum() / m.sum())
                base = dict(hyp=ver if part == "C_calm" else ver + "_S", A="W5G", B="X1", noise=w, subset=part, n_hist=s["n"], share_of_pairs=share,
                            Q_B=s["qb"], sigma_LR=s["sigma"], delta_hist_pct=s["delta_pct"], dm_hist=s["dm"])
                if part == "S_shock":
                    rows.append(dict(base, note="часть «страховка»: знак, без p в семействе"))
                    continue
                for dl in (2.0, 5.0, 10.0, s["delta_pct"]):
                    tag = "hist" if dl == s["delta_pct"] else "%+.1f" % dl
                    r = dict(base, effect=tag, effect_pct=dl)
                    if dl <= 0:
                        rows.append(dict(r, note="исторический эффект ≤ 0: мощность не определена"))
                        continue
                    for al, an in ((ALPHA, "a05"), (ALPHA / M_FAM, "holm7")):
                        for p in POWERS:
                            r["n_p%02d_%s" % (int(p * 100), an)] = n_req(s["sigma"], s["qb"], dl, al, p) / share   # в парах всего
                        for hz in HORIZONS:
                            r["pow_n%d_%s" % (hz, an)] = pw(s["sigma"], s["qb"], dl, al, hz * share)
                    rows.append(r)
    # по годам: пары, gap, C, S (v1 и v2), месяцев шока v2
    mon = x["d"].dt.to_period("M").to_numpy()
    for y in sorted(set(yr)):
        m = yr == y
        rows.append(dict(hyp="YEAR", noise=str(y), n_hist=int(m.sum()), n_gap_on=int((m & on).sum()), n_C=int((m & on & (sh == 0)).sum()),
                         n_S=int((m & on & (sh == 1)).sum()), n_e1=int((m & (x["s"].to_numpy() == 1)).sum()),
                         n_C2=int((m & on & (sm == 0)).sum()), n_S2=int((m & on & (sm == 1)).sum()), n_undef2=int((m & on & (sm == -1)).sum()),
                         months_shock2=int(len(set(mon[m & (sm == 1)]))), months2=int(len(set(mon[m])))))
    # проверка нормального приближения бутстрапом (окно M): H1 при −1 % и H7 (v2) при +5 %, n = 2500 пар
    D1, LB1 = L["W5G"] - L["X1"], L["X1"]
    rows.append(dict(hyp="BOOT", A="W5G", B="X1", noise="M", effect_pct=-1.0, n_eval=2500,
                     boot_a05=boot_power(D1, LB1, -1.0, 2500, ALPHA, -1), boot_holm7=boot_power(D1, LB1, -1.0, 2500, ALPHA / M_FAM, -1),
                     norm_a05=pw(stats(D1, LB1)["sigma"], LB1.mean(), -1.0, ALPHA, 2500), norm_holm7=pw(stats(D1, LB1)["sigma"], LB1.mean(), -1.0, ALPHA / M_FAM, 2500)))
    mC = on & (sm == 0)
    DC, LBC = L["W5G"][mC] - L["X1"][mC], L["X1"][mC]
    nC = int(round(2500 * mC.mean()))
    sC = stats(DC, LBC)
    rows.append(dict(hyp="BOOT", A="W5G", B="X1", noise="M", subset="C_calm", effect_pct=5.0, n_eval=2500, n_sub=nC,
                     boot_a05=boot_power(DC, LBC, 5.0, nC, ALPHA, +1), boot_holm7=boot_power(DC, LBC, 5.0, nC, ALPHA / M_FAM, +1),
                     norm_a05=pw(sC["sigma"], sC["qb"], 5.0, ALPHA, nC), norm_holm7=pw(sC["sigma"], sC["qb"], 5.0, ALPHA / M_FAM, nC)))
    df = pd.DataFrame(rows)
    head = ["# FWD2 POWER2 — мощность DM (нормальное приближение, НУ лаг 5) по сохранённым рядам окна M; новых данных нет; DECL.md п. 1–4 записан до счёта; H7 v2 (месяц IMOEX) — после просмотра H7v1 (DECL ↺ v2)",
            "# входы: " + "; ".join("%s %s" % (k, sha16(v)) for k, v in {**C.SRC, **SRC}.items()),
            "# n_p50/n_p80 — пар нового периода (для H7 — всего пар, с учётом доли C в окне M); a05 — α = 0,05; holm7 — α = 0,05/7 (первый шаг Холма)"]
    return "\n".join(head) + "\n" + df.to_csv(index=False, float_format="%.6g")


if __name__ == "__main__":
    text = main()
    if "--check" in sys.argv:
        ok = OUT.exists() and OUT.read_text(encoding="utf-8") == text
        print("POWER2 --check:", "ok" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    OUT.write_text(text, encoding="utf-8")
    print("POWER2 записан:", OUT, hashlib.sha256(text.encode()).hexdigest()[:16])
