"""S-R-SPEC, калибровка условия концентрации на СОХРАНЁННЫХ парах рядов (новые варианты не считаются).
Для пары (кандидат, база) на D_M: d_t = QLIKE_cand - QLIKE_base.
- share1 — доля выигрыша от 22 (1 %) дат с наименьшим d_t: sum(-d top) / sum(-d), при sum(-d) > 0;
- d_no_best — Δ, % без 22 лучших для кандидата дат (односторонняя обрезка);
- d_trim — Δ, % без 22 лучших и 22 худших дат (симметричная обрезка 1 % + 1 %), база процентов — QLIKE базы на тех же датах;
- d_no24 — Δ, % без t+ 2022-02-24; d_no_day — без самого влиятельного дня (max |d_t|).
Запуск из корня: PYTHONPATH=. python3 -W ignore <sp>/s_r/conc.py"""
import numpy as np
import pandas as pd

from models_v2.s16 import x1_w3_core as C
from models_v2.s11.v4_ntp_state import qlike

ROOT = C.ROOT
x = C.load()
rv, tp = x["rv"].to_numpy(float), pd.DatetimeIndex(x["t_plus"])
rd = lambda p: pd.read_parquet(ROOT / p)["F"].to_numpy()
ser = dict(K1=x["K1"].to_numpy(), K2=x["K2"].to_numpy(), K3=x["K3"].to_numpy(), B_mix=x["M"].to_numpy(), X1=x["F"].to_numpy(),
           W5G=rd("data/regen/harj_x1/W5/w5g.parquet"), W8G=rd("data/regen/harj_x1/W8/w8g.parquet"),
           SA2=rd("data/regen/harj_struct/A/a2.parquet"), SA2p=rd("data/regen/harj_struct/A/a2p.parquet"), SCK=rd("data/regen/harj_struct/CM/ck.parquet"))
K = int(np.ceil(0.01 * len(rv)))
i24 = int(np.flatnonzero(tp == pd.Timestamp("2022-02-24"))[0])


def pct(dd, lb, keep):
    return 100 * dd[keep].mean() / lb[keep].mean()


rows = []
for a, b in (("K2", "K1"), ("B_mix", "K1"), ("X1", "K2"), ("X1", "K3"), ("X1", "B_mix"), ("W5G", "X1"), ("W8G", "W5G"), ("W8G", "X1"),
             ("SA2", "W5G"), ("SA2", "X1"), ("SA2", "SA2p"), ("SCK", "W5G"), ("K1", "K2")):
    lb = qlike(rv, ser[b])
    dd = qlike(rv, ser[a]) - lb
    o = np.argsort(dd, kind="stable")
    all_ = np.ones(len(dd), bool)
    nb, tr, n24, nday = all_.copy(), all_.copy(), all_.copy(), all_.copy()
    nb[o[:K]] = False
    tr[o[:K]] = False
    tr[o[-K:]] = False
    n24[i24] = False
    nday[int(np.argmax(np.abs(dd)))] = False
    g = -dd.sum()
    rows.append(dict(cand=a, base=b, d=pct(dd, lb, all_), share1=(-dd[o[:K]].sum() / g) if g > 0 else np.nan,
                     d_no_best=pct(dd, lb, nb), d_trim=pct(dd, lb, tr), d_no24=pct(dd, lb, n24), d_no_day=pct(dd, lb, nday), frac_win=float(np.mean(dd < 0))))
R = pd.DataFrame(rows)
R.to_csv(__file__.replace("conc.py", "conc.csv"), index=False)
print("D_M n = %d, 1 %% = %d дат" % (len(rv), K))
print(R.to_string(index=False, float_format=lambda v: "%+.4f" % v))
