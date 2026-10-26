"""Детектор утечек на позициях С9 v2: σ после t → шум; позиции A·g всех точек D/B/C на барах ≤ t
обязаны совпасть побайтно; самопроверка σ_t := σ_{t+1} обязана упасть. Запуск из worktree:
PYTHONPATH=. python3 -W ignore <этот файл> <out.csv>"""
import sys

import numpy as np
import pandas as pd

from fastbt import s9

s9.use_v2()
s9.load_c_dop()
eng = s9.Engine(s9.load_cache())
RVW0 = eng.rvw().copy()
RVW_CUR = [RVW0]
x = s9.S9.__new__(s9.S9)
x.e, x.reps = eng, 10
rd = eng.dates[eng.r["date"]]
DATES = ("2019-06-28", "2020-06-30", "2021-06-30")


def noise_after(S, t):
    rng = np.random.RandomState(int(t.replace("-", "")))
    X = S.copy()
    tt = np.datetime64(pd.Timestamp(t))
    for ti, tk in enumerate(eng.tickers):
        d = eng.cache[tk]["dates"]
        m = np.zeros(S.shape[1], bool)
        m[:len(d)] = d > tt
        X[ti, m] = np.exp(rng.normal(-3.5, 0.8, m.sum()))
    return X


def lead(S):
    Y = np.full_like(S, np.nan)
    Y[:, :-1] = S[:, 1:]
    return Y


def positions(src, S):
    eng.sig[src] = S
    out = {}
    for fam, pts in x.d_points(src).items():
        for lab, G, _ in pts:
            out[f"D:{fam}:{lab}"] = eng.A_row * eng.g_row(G)
    for cap in (1.0, 1.5, 2.0):
        for band in (0.0, 0.10, 0.20):
            out[f"B:cap{cap:g}_band{int(band*100)}"] = eng.A_row * eng.g_row(x.b_weights(src, cap, band))
    p1, p2 = x.c_points(src)
    fams = [("C1", p1), ("C2", p2)] + s9.c_dop_gates(S, RVW_CUR[0], eng.n_t, s9.C_DOP)
    for fam, pts in fams:
        for lab, G in pts:
            out[f"C:{fam}:{lab}"] = eng.A_row * eng.g_row(G)
    return out


rows = []
for src in ("hyb_h1", "hyb_h5"):
    S0 = eng.sigma(src).copy()
    for leak in (False, True):
        tr = lead if leak else (lambda S: S)
        RVW_CUR[0] = tr(RVW0)
        base = positions(src, tr(S0))
        for t in DATES:
            RVW_CUR[0] = tr(noise_after(RVW0, t))      # √RV_w для C* — тоже шум после t
            pert = positions(src, tr(noise_after(S0, t)))
            RVW_CUR[0] = tr(RVW0)
            m = rd <= pd.Timestamp(t)
            bad = [k for k in base if not np.array_equal(base[k][m], pert[k][m])]
            after = sum(not np.array_equal(base[k][~m], pert[k][~m]) for k in base)
            ok = (len(bad) == 0) if not leak else (len(bad) > 0)
            rows.append(dict(pometka=s9.POMETKA, sigma=src, t=t,
                             proverka="самопроверка σ_t := σ_{t+1} (обязана упасть)" if leak
                             else "будущее не влияет: σ после t → шум",
                             n_pozicij=len(base), n_rashozhdeniy_do_t=len(bad),
                             n_izmenilis_posle_t=after, ok=ok, primer=", ".join(bad[:3])))
            print(rows[-1], flush=True)
    eng.sig[src] = S0
bad_w = 0
from fastbt.data import year_bounds
for ti, tk in enumerate(eng.tickers):
    d = eng.cache[tk]["dates"]
    for Y in s9.YEARS:
        vs, ve = eng.vwin[(ti, Y)]
        if not (pd.Timestamp(d[ve - 1]) < pd.Timestamp(f"{Y}-01-01") <= pd.Timestamp(d[ve])
                and ve == year_bounds(d, Y)[0]):
            bad_w += 1
rows.append(dict(pometka=s9.POMETKA, sigma="—", t="2018–2021",
                 proverka="окна валидации/калибровки кончаются на _year_bounds(Y)[0]",
                 n_pozicij=len(eng.tickers) * len(s9.YEARS), n_rashozhdeniy_do_t=bad_w, ok=bad_w == 0))
df = pd.DataFrame(rows)
df.to_csv(sys.argv[1], sep=";", decimal=",", index=False, encoding="utf-8-sig")
print(f"детектор: {int(df['ok'].sum())} из {len(df)} OK")
