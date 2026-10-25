"""Единственный тестовый прогон главы 4 (2022–2026) по зафиксированному дизайну стратегий.

python3 -m fastbt.final_test --main fastbt/designs/final/main.yaml \
    --extra d_binary=fastbt/designs/final/d_binary.yaml d_cont=fastbt/designs/final/d_cont.yaml \
    --harj-vpred <σ_pred HAR-J (h=1)> --boot-reps 10000 --workers 10 [--val-start 2017]
python3 -m fastbt.final_test ... --scope dev        (проверка кода на 2018–2021, без замка)

Замок: --scope test без docs/otchety/FINAL_DESIGN_strategies.md падает (fastbt.dev.check_scope).
Считает: позиции A/B/C/D основного дизайна и дополнительных; МСП по окнам и годам; ΔШарп
B/C/D − A (парный бутстрап, seed 42); ΔCE (γ = 3, 5, 10) с бутстрапом и K_min = ΔCost / ΔCE;
C на гибридном σ_pred против C на σ_pred HAR-J (тот же дизайн, другой σ); H3 (бенчмарки
fastbt.bench) для каждого подхода; ряды для рисунков (fastbt.export).
Выход (только --scope test): docs/tables/tabl_4_final_*.csv; позиции и ряды —
<основная копия>/data/fastbt/final_<scope>_*.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "docs" / "tables"
GAMMAS = (3, 5, 10)
COSTS_MLN = (1.0, 3.3, 5.0, 13.1, 16.8)       # ΔCost, млн руб. в год (как в FINAL_RESULTS.md)
BPY = 252


def ce(r, gamma):
    w = np.maximum(1.0 + r, 1e-10)
    return np.mean(w ** (1 - gamma), axis=-1) ** (1 / (1 - gamma)) - 1


def ce_boot(a, b, gamma, reps, seed=42):
    """ΔCE(a) − CE(b), % годовых (FB:38-55); p = доля Δ* ≤ 0; ДИ 95%."""
    obs = (ce(a, gamma) - ce(b, gamma)) * BPY * 100
    rng = np.random.RandomState(seed)
    n = len(a)
    d = np.empty(reps)
    for i0 in range(0, reps, 500):
        k = min(500, reps - i0)
        idx = rng.randint(0, n, (k, n))
        d[i0:i0 + k] = (ce(a[idx], gamma) - ce(b[idx], gamma)) * BPY * 100
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(obs), float(np.mean(d <= 0)), float(lo), float(hi)


def run_design(path, scope, val_start, vpred, workers, label):
    from fastbt import dev
    from fastbt.cache import MAIN
    from fastbt.data import TICKERS
    from fastbt.pipeline import run_all
    design, _ = dev.load_design(path)
    vp = vpred or design.get("vpred")
    os.environ["FASTBT_VPRED"] = str(vp)
    cfg = dev.make_config(design, scope)
    cfg = dataclasses.replace(cfg, val_start=val_start if scope == "test" else cfg.val_start)
    vmax = pd.read_parquet(vp, columns=["date"])["date"].max()
    if scope == "test" and pd.Timestamp(vmax) < pd.Timestamp("2026-01-01"):
        cfg = dataclasses.replace(cfg, years_bcd=tuple(y for y in cfg.years_bcd if y <= 2025))
        print(f"[{label}] σ_pred до {pd.Timestamp(vmax).date()}: B/C/D только 2022–2025")
    pos, metas = run_all(cfg, design.get("tickers", TICKERS),
                         design.get("sids", ["S1", "S2", "S3", "S4", "S5", "S6"]), workers,
                         log=lambda m: None)
    out = MAIN / "data" / "fastbt" / f"final_{scope}_{label}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pos.to_parquet(out, index=False)
    return pos, cfg, dev.choices(metas, set(cfg.years_bcd))


def windows(scope):
    if scope == "test":
        return {"2022–2025": ("2022-01-01", "2025-12-31"), "2022–2026": ("2022-01-01", "2026-12-31")}
    return {"2018–2021": ("2018-01-01", "2021-12-31")}


def tables(pos, cfg, scope, reps, label):
    from fastbt import bench, msp
    from fastbt.report import portfolios, summary
    years = list(cfg.years_a) if scope == "test" else list(cfg.years_bcd)
    ports = portfolios(pos, cfg, years)
    rate = msp.rate_cbr() if cfg.fix_rate else msp.rate_restored()
    m_rows, ce_rows, h3 = [], [], []
    for ap, p in ports.items():
        for w, (s, e) in windows(scope).items():
            r = dict(design=label, approach=ap, window=w, **msp.metrics(p.loc[s:e]))
            for y in years:
                ex = p.loc[p.index.year == y, "excess"]
                r[f"sharpe_{y}"] = msp.sharpe(ex) if len(ex) > 2 else np.nan
            m_rows.append(r)
            if ap != "A" and "A" in ports:
                j = p.loc[s:e].index.intersection(ports["A"].loc[s:e].index)
                a_, b_ = p.loc[j, "total"].values, ports["A"].loc[j, "total"].values
                for g in GAMMAS:
                    obs, pv, lo, hi = ce_boot(a_, b_, g, reps)
                    kmin = {f"kmin_{c:g}_mln": (c / (obs / 100) if obs > 0 else np.inf)
                            for c in COSTS_MLN}
                    ce_rows.append(dict(design=label, delta=f"{ap}-A", window=w, gamma=g,
                                        d_ce_pct=obs, p=pv, ci_lo=lo, ci_hi=hi, **kmin))
            if scope == "test":
                t = bench.h3_table(p, rate, reps, s, e).assign(design=label, approach=ap, window=w)
                t["h3_accept"] = np.where(t["p_boot"].isna(), "",
                                          np.where((t["d_sharpe_msp_minus"] > 0)
                                                   & (t["p_boot"] < 0.05), "да", "нет"))
                h3.append(t)
    _, deltas = summary(pos, cfg, years, reps)
    deltas = deltas.assign(design=label)
    return (pd.DataFrame(m_rows), deltas, pd.DataFrame(ce_rows),
            pd.concat(h3, ignore_index=True) if h3 else pd.DataFrame(), ports)


def hyb_vs_harj(ports_h, ports_j, scope, reps):
    from fastbt.report import boot_delta
    rows = []
    for ap in ("B", "C", "D"):
        if ap not in ports_h or ap not in ports_j:
            continue
        for w, (s, e) in windows(scope).items():
            ph, pj = ports_h[ap].loc[s:e], ports_j[ap].loc[s:e]
            j = ph.index.intersection(pj.index)
            obs, pv, lo, hi = boot_delta(ph.loc[j, "excess"].values, pj.loc[j, "excess"].values, reps)
            r = dict(approach=ap, window=w, d_sharpe_hyb_minus_harj=obs, p=pv, ci_lo=lo, ci_hi=hi,
                     days=len(j))
            for g in GAMMAS:
                r[f"d_ce_g{g}"], r[f"p_ce_g{g}"], _, _ = ce_boot(
                    ph.loc[j, "total"].values, pj.loc[j, "total"].values, g, reps)
            rows.append(r)
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--main", required=True)
    p.add_argument("--extra", nargs="*", default=[], help="метка=путь.yaml")
    p.add_argument("--harj-vpred", required=True)
    p.add_argument("--scope", choices=["dev", "test"], default="test")
    p.add_argument("--val-start", type=int, default=2017)
    p.add_argument("--boot-reps", type=int, default=10000)
    p.add_argument("--workers", type=int, default=10)
    a = p.parse_args(argv)
    from fastbt import dev
    from fastbt.cache import MAIN
    dev.check_scope(a.scope)
    if not Path(a.harj_vpred).exists():
        sys.exit(f"нет σ_pred HAR-J: {a.harj_vpred}")
    t0 = time.time()
    designs = [("main", a.main)] + [tuple(x.split("=", 1)) for x in a.extra]
    allm, alld, allce, allh3, chosen = [], [], [], [], []
    ports_main = None
    for label, path in designs:
        pos, cfg, ch = run_design(path, a.scope, a.val_start, None, a.workers, label)
        m, d, c, h, ports = tables(pos, cfg, a.scope, a.boot_reps, label)
        allm.append(m), alld.append(d), allce.append(c), allh3.append(h)
        chosen.append(dict(design=label, path=path, chosen=ch, years_bcd=str(cfg.years_bcd),
                           val_start=cfg.val_start))
        if label == "main":
            ports_main, cfg_main, pos_main = ports, cfg, pos
        print(f"[{label}] готово, {time.time() - t0:.0f} с")
    pos_j, cfg_j, _ = run_design(a.main, a.scope, a.val_start, a.harj_vpred, a.workers, "main_harj")
    from fastbt.report import portfolios
    yrs = list(cfg_j.years_a) if a.scope == "test" else list(cfg_j.years_bcd)
    hj = hyb_vs_harj(ports_main, portfolios(pos_j, cfg_j, yrs), a.scope, a.boot_reps)
    out = {"tabl_4_final_msp.csv": pd.concat(allm), "tabl_4_final_delta.csv": pd.concat(alld),
           "tabl_4_final_ce_kmin.csv": pd.concat(allce), "tabl_4_final_h3.csv": pd.concat(allh3),
           "tabl_4_final_c_gibrid_vs_harj.csv": hj, "tabl_4_final_vybor.csv": pd.DataFrame(chosen)}
    pd.set_option("display.width", 250)
    for k, v in out.items():
        print(f"== {k}\n{v.round(3).head(40).to_string(index=False)}")
    if a.scope == "test":
        TABLES.mkdir(parents=True, exist_ok=True)
        for k, v in out.items():
            v.to_csv(TABLES / k, index=False)
        from fastbt import export, msp
        series = MAIN / "data" / "fastbt" / "final_series" / "perstrat_daily_returns"
        export.export(pos_main, series, msp.rate_cbr())
        print(f"ряды для рисунков → {series}")
    print(f"всего {time.time() - t0:.0f} с")
    return 0


if __name__ == "__main__":
    sys.exit(main())
