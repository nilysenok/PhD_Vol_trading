"""V6 лист Z3, диагностики 4–7: калибровка Mincer–Zarnowitz (WHY_MZ), доля бустинговой поправки (WHY_CAPTURE), устойчивость во времени (WHY_ROLL), величина поправки (WHY_GSIZE).
Окна — по году цели t⁺ (`v6_0_q.wmask`). MZ: ln RV = a + b ln F, HAC лаг 5 (`v6_why_stat.hac_ols`). Доли — в процентах."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_q as Q, v6_why_data as DT, v6_why_stat as ST

MZ_WINS = ("2017_2021", "2022_2025")
CAP_WINS = ("2017_2021", "2022_2025", "all")
ROLL = 250
CLIP = 1.0 - 1e-9


def mz_table(d: dict) -> pd.DataFrame:
    rv, rows = d["rv"], []
    lr = np.log(rv)
    for s in DT.SER:
        lf = np.log(d["f"][s])
        for w in MZ_WINS:
            m = Q.wmask(d["tp"], w)
            beta, se = ST.hac_ols(lr[m], np.column_stack([np.ones(m.sum()), lf[m]]))
            rows.append(dict(series=s, kind="mz", window=w, n=int(m.sum()), a=beta[0], se_a=se[0], b=beta[1], se_b=se[1], t_b_vs_1=(beta[1] - 1) / se[1] if se[1] > 0 else np.nan, shift=float(np.mean(lr[m] - lf[m]))))
        for y, m in ST.year_masks(d["tp"]).items():
            rows.append(dict(series=s, kind="year", window=y, n=int(m.sum()), shift=float(np.mean(lr[m] - lf[m]))))
    return pd.DataFrame(rows)


def capture_table(d: dict) -> pd.DataFrame:
    f, rows = d["f"], []
    for base in DT.BOOST[::-1]:
        x_all = np.log(f[base] / f["K1"])
        for h in DT.HYB:
            y_all = np.log(f[h] / f["K1"])
            for w in CAP_WINS:
                m = Q.wmask(d["tp"], w)
                x, y = x_all[m], y_all[m]
                vx = float(x.var())
                rows.append(dict(base=base, hybrid=h, window=w, n=int(m.sum()), slope=float(((x - x.mean()) * (y - y.mean())).mean()) / vx if vx > 0 else np.nan,
                                 corr=float(np.corrcoef(x, y)[0, 1]) if vx > 0 and float(y.var()) > 0 else np.nan, same_sign_pct=100.0 * float((np.sign(x) == np.sign(y)).mean())))
    return pd.DataFrame(rows)


def roll_means(x, k1q, win: int = ROLL) -> np.ndarray:
    """Скользящая Δ в % уровня K1 того же окна: окна из win смежных дат, шаг 1."""
    cx, ck = np.concatenate([[0.0], np.cumsum(x)]), np.concatenate([[0.0], np.cumsum(k1q)])
    return 100.0 * (cx[win:] - cx[:-win]) / (ck[win:] - ck[:-win])


def roll_table(d: dict) -> pd.DataFrame:
    rv, f, rows = d["rv"], d["f"], []
    k1q = Q.ql(rv, f["K1"])
    m1, m2 = Q.wmask(d["tp"], "2017_2021"), Q.wmask(d["tp"], "2022_2025")
    for h, b in DT.PAIRS:
        x = ST.dd(rv, f[h], f[b])
        r = roll_means(x, k1q)
        rows.append(dict(hybrid=h, base=b, kind="roll", n_windows=len(r), share_neg_pct=100.0 * float((r < 0).mean()), min_pct=float(r.min()), max_pct=float(r.max())))
        rows.append(dict(hybrid=h, base=b, kind="split", n_early=int(m1.sum()), n_late=int(m2.sum()), **ST.diff_ci(x[m1], float(k1q[m1].mean()), x[m2], float(k1q[m2].mean()))))
    return pd.DataFrame(rows)


def gsize_table(d: dict) -> pd.DataFrame:
    f, rows = d["f"], []
    masks = {**ST.year_masks(d["tp"]), "all": np.ones(len(d["rv"]), bool)}
    for s in DT.HYB + DT.BOOST[::-1]:
        a = np.abs(np.log(f[s] / f["K1"]))
        for w, m in masks.items():
            if m.any():
                rows.append(dict(series=s, window=w, n=int(m.sum()), mean_abs=float(a[m].mean()), p95_abs=float(np.percentile(a[m], 95)), at_clip_pct=100.0 * float((a[m] >= CLIP).mean())))
    return pd.DataFrame(rows)
