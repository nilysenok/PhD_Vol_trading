"""Проверки S-C-IMPL-1 (`cm_feat`): --check (F.G1), --check-k7 (F.G2), --check-pit (F.G3), --write. Только чтение, кроме --write."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from models_v2 import har as H
from models_v2.s10 import hp_ctrl as HC
from models_v2.s18 import c_data, c_decomp
from models_v2.s18 import cm_feat as F
from models_v2.s17.fwd_src import Src


def dir_hash(d=F.CM_DIR) -> str:
    h = hashlib.sha256()
    for p in sorted(d.glob("*")) if d.exists() else []:
        h.update(p.name.encode() + (p.read_bytes() if p.is_file() else b""))
    return h.hexdigest()


def check() -> list:
    h0, bad = dir_hash(), []
    tab = F.load_c()
    k = F.components(tab)
    n, first = len(tab), int(np.argmax(tab["n_cov"].to_numpy() > 0))
    ident = np.abs(tab["rv_common"] + tab["rv_specific"] + tab["rv_uncovered"] - tab["rv"]) / tab["rv"]
    print("сессий %d; первая покрытая %s; тождество c+p+u=RV: max отн. %.2e" % (n, tab.index[first].date(), ident.max()))
    bad += [] if n == 3124 and str(tab.index[first].date()) == "2014-08-27" and ident.max() <= 1e-13 else ["тождество / сетка"]
    bad += [] if np.isnan(k["c"][:first]).all() and np.isfinite(k["c"][first:]).all() else ["NaN до первой покрытой (Д4)"]
    f = F.build(tab=tab)
    dm = F.dm_dates()
    s = f.loc[dm]
    print("D_M: %d дат" % len(dm))
    for pre, cols in (("", F.COLS_REAL), ("L_", F.COLS_LAG)):
        z = s[cols]
        bad += [] if np.isfinite(z.to_numpy()).all() else ["%s: нефинитные входы на D_M" % (pre or "real")]
        q = s[[pre + c for c in ("c", "p", "c_m", "p_m", "Hc", "Hp")]]
        bad += [] if (q > 0).all().all() else ["%s: X ≤ 0" % (pre or "real")]
        sh = s[[pre + c for c in ("sc", "sp", "su")]]
        sm = sh.sum(axis=1)
        print("%s ŝ: min %.3g max %.3g; max|Σŝ − 1| = %.2e" % (pre or "real", sh.min().min(), sh.max().max(), (sm - 1).abs().max()))
        bad += [] if sh.min().min() >= 0 and sh.max().max() <= 1 and (sm - 1).abs().max() <= 1e-12 else ["%s: ŝ вне [0, 1] или Σ ≠ 1" % (pre or "real")]
    bad += [] if dir_hash() == h0 else ["выходы изменены --check"]
    return bad


def check_k7() -> list:
    from models_v2.s16 import x1_w3_core as C
    tab = F.load_c()
    k = F.components(tab)
    src = Src()
    from models_v2 import daily as D
    dd, _ = D.build_daily(src.bars("SBER", F.LAST_SESSION), index_days=src.index_days(F.LAST_SESSION))
    seg, j, bad = dd["seg"].to_numpy(), dd["j"].to_numpy(float), []
    x = C.load()
    rows = dd.index.get_indexer(pd.DatetimeIndex(x["d"]))
    pos = rows[:: max(1, len(rows) // 60)]
    for name in ("c", "p"):
        a = F.harj_hat(k[name], seg, j, rows[::1])
        z = k[name]
        pan = H.make_panel(pd.DataFrame({"rv": z, "rv_w": F.DC.wmean(z, seg, 5), "rv_m": F.DC.wmean(z, seg, 22), "j": j, "seg": seg}, index=dd.index))
        b = np.array([HC.harj_at(pan, int(i)) for i in pos])
        d = np.nanmax(np.abs(a[pos] - b))
        print("Ĥ_%s: %d строк, max |Ĥ − harj_at| = %.2e; конечно на %d из %d" % (name, len(pos), d, np.isfinite(a[rows]).sum(), len(rows)))
        bad += [] if d == 0 and np.isfinite(a[rows]).all() else ["Ĥ_%s ≠ harj_at побитно или не конечен" % name]
    h = F.harj_hat(dd["rv"].to_numpy(float), seg, j, rows)[rows]
    k1 = x["K1"].to_numpy(float)
    diff = np.abs(h - k1)
    print("X := RV: max |Ĥ − K1| = %.2e (побитно: %d из %d), max отн. %.2e" % (diff.max(), int((diff == 0).sum()), len(rows), (diff / k1).max()))
    bad += [] if diff.max() == 0 else ["X := RV не даёт K1 побитно"]
    return bad


def check_pit(n_days: int = 10, n_sess: int = 3) -> list:
    tab = F.load_c()
    base = F.build(tab=tab)
    dm = F.dm_dates()
    rng = np.random.default_rng(0)
    bad = []
    for d in rng.choice(dm.to_numpy(), n_days, replace=False):
        i = int(tab.index.get_loc(pd.Timestamp(d)))
        t2 = tab.copy()
        for col in ("rv", "rv_common", "rv_specific", "rv_uncovered"):
            t2.iloc[i + 1:, t2.columns.get_loc(col)] = rng.uniform(1.0, 1e3, len(t2) - i - 1)
        g = F.build(tab=t2, rows=[i])
        a, b = base.iloc[i][F.COLS_REAL + F.COLS_LAG].to_numpy(float), g.iloc[i][F.COLS_REAL + F.COLS_LAG].to_numpy(float)
        bad += [] if np.array_equal(a, b, equal_nan=True) else ["%s: порча C сессий > d меняет входы" % pd.Timestamp(d).date()]
    print("порча C сессий > d: %d дат" % n_days)
    first = int(np.argmax(tab["n_cov"].to_numpy() > 0))
    ok = np.isnan(base["L_c"].to_numpy()[: first + F.SHIFT]).all() and np.isfinite(base["L_c"].to_numpy()[first + F.SHIFT:]).all()
    bad += [] if ok else ["S-CKL: s − 252 < первой покрытой не NaN"]
    bad += [] if "SBER" not in c_data.POOL else ["SBER в факторе"]
    days = [pd.Timestamp(d) for d in rng.choice(dm[dm >= "2019-01-01"].to_numpy(), n_sess, replace=False)]
    for d in days:
        inp = c_data.build_inputs(cut=d)
        s = int(np.flatnonzero(inp.days == d)[0])
        r = [x for x in c_decomp.session_rows(inp, s) if x["k_factors"] == 1][0]
        t = tab.loc[d]
        same = all(abs(r[k] - t[k]) <= 1e-12 * t["rv"] for k in ("rv_common", "rv_specific", "rv_uncovered"))
        print("пересчёт C на cut = %s: %s" % (d.date(), "совпал" if same else "РАСХОДИТСЯ"))
        bad += [] if same else ["пересчёт сессии %s при cut = d расходится с C_sessions" % d.date()]
    return bad


def main(argv) -> int:
    if "--write" in argv:
        print("feat.parquet: %d строк" % len(F.write()))
        return 0
    key = "--check-k7" if "--check-k7" in argv else "--check-pit" if "--check-pit" in argv else "--check" if "--check" in argv else None
    if key is None:
        raise SystemExit("укажите --check, --check-k7, --check-pit или --write")
    bad = {"--check": check, "--check-k7": check_k7, "--check-pit": check_pit}[key]()
    print("%s: %s" % (key, "ok" if not bad else "FAIL " + "; ".join(bad)))
    return 1 if bad else 0
