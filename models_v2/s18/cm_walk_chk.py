"""Проверки S-C-IMPL-3 (`cm_walk`): --check (W.G4), --check-pit (W.G3). Только чтение."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s16 import x1_ya_walk as YA
from models_v2.s18 import cm_feat as CF
from models_v2.s18 import cm_model as M
from models_v2.s18 import cm_walk as W
from models_v2.s18.cm_feat_chk import dir_hash

MONTH = "2023-02-01"


def _same(a: dict, b: dict, keys) -> bool:
    return all(np.array_equal(np.asarray(a[k]), np.asarray(b[k]), equal_nan=True) for k in keys)


def check() -> list:
    h0, bad = dir_hash(W.CM_DIR), []
    x, gap, D0 = W.inputs("CK")
    xd = pd.DatetimeIndex(x["d"])
    starts = YA.month_starts(xd)
    act = [p for p in starts if xd[p] >= YA.ACTIVATION]
    print("D_M: %d дат; месяцев %d; включение %s; активных переоценок %d" % (len(x), len(starts), xd[act[0]].date(), len(act)))
    bad += [] if len(x) == 2198 and len(starts) == 106 and str(xd[act[0]].date()) == "2018-07-02" else ["D_M / месяцы / включение"]
    feat = CF.load_feat()
    a = int(np.flatnonzero(xd == pd.Timestamp(MONTH))[0])
    b = starts[starts.index(a) + 1]
    for var, name in W.OUTS.items():
        p = W.CM_DIR / name
        if not p.exists():
            bad.append("%s: нет файла" % name)
            continue
        df = pd.read_parquet(p)
        ok = len(df) == 2198 and (pd.DatetimeIndex(df["d"]) == xd).all()
        pre = ~df["active"].to_numpy()
        ok = ok and pre.sum() == starts[starts.index(act[0])] and np.array_equal(df["C"].to_numpy()[pre], df["cfg8"].to_numpy()[pre]) and (df["M"].to_numpy()[pre] == 1.0).all()
        lam = df["lam"].dropna().unique()
        ok = ok and set(lam) <= set(M.LAMS) and (df.loc[df["n_train"] < W.MIN_VALID, "lam"].dropna() == W.LAM_DEF).all()
        ok = ok and df["refit_date"].dropna().nunique() == len(act)
        ident = 0.0 if var == "CD" else float(np.max(np.abs((df["F_c"] + df["F_p"] + df["F_u"]) / df["C"] - 1)))
        ok = ok and ident <= 1e-13
        D = D0 if var != "CKL" else W.inputs("CKL", feat)[2]
        r = W.fit_month(D, YA.closed_pool(x["t_plus"], xd, xd[a]), np.arange(a, b), var)
        rec = _same(r, dict(M=df["M"].to_numpy()[a:b]), ("M",))
        print("%s: строк %d; до включения %d (C = cfg8 побитно); λ %s; ΣF = C %.1e; пересчёт месяца %s: %s" % (var, len(df), pre.sum(), sorted(lam), ident, MONTH, "побитно" if rec else "РАСХОДИТСЯ"))
        bad += [] if ok and rec else ["%s: структура / пересчёт месяца" % var]
    bad += [] if dir_hash(W.CM_DIR) == h0 else ["выходы изменены --check"]
    return bad


def _spoil(tab: pd.DataFrame, after, seed: int) -> pd.DataFrame:
    t, g = tab.copy(), np.random.default_rng(seed)
    m = t.index > after
    for c in ("rv", "rv_common", "rv_specific", "rv_uncovered"):
        t.loc[m, c] = g.uniform(1.0, 1e3, int(m.sum()))
    return t


def check_pit() -> list:
    x, gap, D0 = W.inputs("CK")
    xd, tp = pd.DatetimeIndex(x["d"]), pd.DatetimeIndex(x["t_plus"])
    tab, bad = CF.load_c(), []
    starts = YA.month_starts(xd)
    act = [p for p in starts if xd[p] >= pd.Timestamp("2019-01-01")]
    for a in np.random.default_rng(0).choice(act[:-1], 3, replace=False):
        b = starts[starts.index(int(a)) + 1]
        D, pool, days, last = xd[a], YA.closed_pool(tp, xd, xd[a]), np.arange(a, b), xd[b - 1]
        bad_rows = np.flatnonzero(tp > D)
        x2 = x.copy()
        x2.loc[x2.index[bad_rows], "rv"] = np.random.default_rng(1).uniform(1.0, 1e3, len(bad_rows))
        fa = CF.build(tab=_spoil(tab, D, 2))                    # порча C сессий > D: параметры пула не меняются
        fb = CF.build(tab=_spoil(tab, last, 3))                 # порча сессий > последнего дня месяца и RV пар t⁺ > D: параметры и прогнозы месяца
        for var in ("CK", "CKA"):
            base = W.fit_month(D0, pool, days, var)
            ta = W.fit_month(M.build_data(x, fa), pool, days, var)
            tb = W.fit_month(M.build_data(x2, fb), pool, days, var)
            ok_a, ok_b = _same(base, ta, ("theta", "lam", "seed")), _same(base, tb, ("theta", "lam", "seed", "M", "Fc", "Fp", "Fu"))
            print("%s %s: параметры при порче > D: %s; параметры и прогнозы при порче RV t⁺ > D и сессий > конца месяца: %s" % (var, D.date(), "побитно" if ok_a else "РАЗЛИЧАЮТСЯ", "побитно" if ok_b else "РАЗЛИЧАЮТСЯ"))
            bad += [] if ok_a and ok_b else ["%s %s: порча будущего меняет результат" % (var, D.date())]
    return bad
