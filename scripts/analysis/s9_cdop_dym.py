#!/usr/bin/env python3
"""СC, Ъ 22:30: дымовой прогон новых точек сетки C С9 (C2W, C*1, C*2; fastbt/designs/s9/setka_C.yaml)
на 2–3 тикерах и выровненных σ из data/regen/ch4_dev/ + детектор утечек на позициях новых точек.
Полный прогон С9 ведёт СD (fastbt.s9 run --c-dop). Пометка — POMETKA ниже.

python3 scripts/analysis/s9_cdop_dym.py [--tickers SBER,LKOH,MTSS] [--reps 200]
Вход A: кэш С9 (dev) с позицией A основного дизайна (A от σ не зависит), путь --a-cache.
Выход: docs/tables/strat_dev_v2/C_dop_dym.csv, C_dop_detektor.csv.
"""
import argparse
import hashlib
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fastbt import s9  # noqa: E402
from fastbt.cache import MAIN  # noqa: E402

CH4 = MAIN / "data" / "regen" / "ch4_dev"
OUTT = ROOT / "docs" / "tables" / "strat_dev_v2"
SIG = {"hyb_h1": ("sigma_hyb_h1_dev.parquet", "sigma_pred_h1", "h1"),
       "hyb_h5": ("sigma_hyb_h5_dev.parquet", "sigma_pred_h5", "h5"),
       "naive_h1": ("sigma_naive_dev.parquet", "sigma_pred_h1", "h1")}
DET_DATES = ("2019-06-28", "2020-03-16", "2021-06-30")
POMETKA = "после аудита (Н1–Н5, H3, ключи кэшей); дымовой прогон, 3 тикера"   # ЭТ, 26.10


def check_sha():
    ref = {}
    for ln in (CH4 / "SHA256.txt").read_text().splitlines():
        p = ln.split()
        if len(p) >= 2 and p[1].startswith("data/regen/ch4_dev/"):
            ref[Path(p[1]).name] = p[0]
    for fn, _, _ in SIG.values():
        h = hashlib.sha256((CH4 / fn).read_bytes()).hexdigest()
        if ref.get(fn) != h:
            raise SystemExit(f"SHA {fn}: {h} ≠ {ref.get(fn)}")


def shift1(S, n_t):
    Z = np.full_like(S, np.nan)
    for ti in range(S.shape[0]):
        Z[ti, :n_t[ti] - 1] = S[ti, 1:n_t[ti]]
    return Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="SBER,LKOH,MTSS")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--a-cache", default=str(MAIN / "data/fastbt/s9/cache_89c0dd6160.pkl"))
    a = ap.parse_args()
    t0 = time.time()
    check_sha()
    s9.SOURCES.clear()                             # выровненные σ S0 — локальный override
    for lab, (fn, col, h) in SIG.items():
        s9.SOURCES[lab] = (CH4 / fn, col, h)
    spec = s9.load_c_dop()
    full = pickle.loads(Path(a.a_cache).read_bytes())
    tk = a.tickers.split(",")
    cache = {}
    for t in tk:                                   # C — заглушка (= A): строки C основного дизайна
        c = full[t]                                # в дымовом прогоне не смотрятся
        cache[t] = dict(dates=c["dates"], lr=c["lr"], A=c["A"],
                        C={lab: {sid: {Y: c["A"][sid][Y] for Y in s9.YEARS}
                                 for sid in s9.SIDS} for lab in SIG})
    del full
    e = s9.Engine(cache)
    x = s9.S9(e, a.reps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df, _ = x.run_c(tuple(SIG))
    new = df[df["family"].str.match(r"^(C2W|C\*1|C\*2)")].copy()
    new["tickers"] = a.tickers
    new["pometka"] = POMETKA
    OUTT.mkdir(parents=True, exist_ok=True)
    new.to_csv(OUTT / "C_dop_dym.csv", sep=";", decimal=",", index=False, float_format="%.4f",
               encoding="utf-8-sig")
    # детектор на позициях новых точек
    rows = []
    row_date = e.dates[e.r["date"]]
    RV = e.rvw()
    for src in ("hyb_h1", "hyb_h5", "naive_h1"):
        S = e.sigma(src)
        rv = None if src.startswith("naive") else RV
        for ctl in ("как есть", "контроль: сдвиг σ +1"):
            f = (lambda M: M) if ctl == "как есть" else (lambda M: shift1(M, e.n_t))
            base = s9.c_dop_gates(f(S), None if rv is None else f(rv), e.n_t, spec)
            pos0 = {(fm, lab): e.A_row * e.g_row(G) for fm, pts in base for lab, G in pts}
            hit = {k: False for k in pos0}
            ok = {k: True for k in pos0}
            for i, t in enumerate(DET_DATES):
                tt = np.datetime64(t)
                rng = np.random.RandomState(42 + i)
                Sn, RVn = S.copy(), None if rv is None else rv.copy()
                for ti, tkr in enumerate(e.tickers):
                    m = np.zeros(e.nmax, bool)
                    m[:e.n_t[ti]] = e.cache[tkr]["dates"] > tt
                    Sn[ti, m] = np.exp(rng.normal(-4, 1, m.sum()))
                    if RVn is not None:
                        RVn[ti, m] = np.exp(rng.normal(-4, 1, m.sum()))
                pert = s9.c_dop_gates(f(Sn), None if RVn is None else f(RVn), e.n_t, spec)
                past = row_date <= tt
                for fm, pts in pert:
                    for lab, G in pts:
                        p1 = e.A_row * e.g_row(G)
                        same = np.array_equal(pos0[(fm, lab)][past], p1[past])
                        ok[(fm, lab)] &= same
                        hit[(fm, lab)] |= not same
            for (fm, lab) in pos0:
                v = ("OK" if ok[(fm, lab)] else "УТЕЧКА") if ctl == "как есть" else \
                    ("поймана" if hit[(fm, lab)] else "НЕ ПОЙМАНА")
                rows.append(dict(family=fm, point=lab, sigma=src, variant=ctl,
                                 dates=",".join(DET_DATES), verdict=v, pometka=POMETKA))
    det = pd.DataFrame(rows)
    det.to_csv(OUTT / "C_dop_detektor.csv", sep=";", index=False, encoding="utf-8-sig")
    print(f"новых строк {len(new)} (точек {int((new['point'] != 'семья').sum())}), "
          f"детектор: {det.groupby(['variant', 'verdict']).size().to_dict()}, "
          f"{time.time() - t0:.0f} с")
    print(new[["family", "point", "h", "sigma", "sharpe", "sharpe_A", "p_vs_A", "win"]]
          .head(60).to_string())


if __name__ == "__main__":
    main()
