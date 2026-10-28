"""Досчёт к поручению владельца 26.10 09:40 (только разработка 2018–2021, σ_pred final3, кэш PR #99).
1) p Ледуа–Вольфа (блок 5, 1 000) для Δ к A у ключевых строк; 2) поправка на множественный выбор
по 26 точкам C (одношаговый max-тест Романо–Вольфа / White RC, IID по дням, seed 42, 1 000);
3) max|FINAL − final3| на 2023–2024. Тест (после 2021) не читается: σ обрезана s9.CUT.
Запуск: PYTHONPATH=. python3 scripts/s9_dop_0940.py  (таблицы не пишет)."""
import os
import tempfile

import numpy as np
import pandas as pd

os.environ.setdefault("S9_TABLES", tempfile.mkdtemp())
from fastbt import s9  # noqa: E402
from fastbt.report import _sh  # noqa: E402
from fastbt.s9_stats import lw_sharpe_diff  # noqa: E402

s9.use_sigma_h1("final3")
eng = s9.Engine(s9.load_cache())
x = s9.S9(eng, 1000)
dfc, cp = x.run_c(("hyb_h1", "hyb_h5"))
dfd, dp = x.run_d(("hyb_h1",))
A = x.A["excess"].values
rows = []
keys = [(("Cref", "hyb_h1"), "C u02 h1"), (("Cref", "hyb_h5"), "C u02 h5"),
        (("C1:W504_p70", "hyb_h5"), "C1 W504 p70 h5"), (("C1:W504_p70", "hyb_h1"), "C1 W504 p70 h1"),
        (("C1 семья", "hyb_h5"), "C1 семья h5"), (("C1 семья", "hyb_h1"), "C1 семья h1")]
for k, nm in keys:
    p = cp[k][0]["excess"].values
    d, se, pv = lw_sharpe_diff(p, A, 1000, 5)
    _, pi = x.boot(cp[k][0], x.A)
    rows.append(dict(design=nm, d=d, p_lw5=pv, p_iid=pi))
for fam in ("D6", "D6lag1"):
    p = dp[(fam, "hyb_h1")][0]
    d, se, pv = lw_sharpe_diff(p["excess"].values, A, 1000, 5)
    _, pi = x.boot(p, x.A)
    rows.append(dict(design=f"{fam} семья h1", d=d, p_lw5=pv, p_iid=pi))
print(pd.DataFrame(rows).round(4).to_string(index=False))

# max-тест по 26 точкам C (u02 + C1 9 + C2 3, × h1/h5)
pts = [k for k in cp if k[0] not in ("C1 семья", "C2 семья")]
M = np.stack([cp[k][0]["excess"].values for k in pts])
n = M.shape[1]
obs = np.array([_sh(M[i]) - _sh(A) for i in range(len(pts))])
rng = np.random.RandomState(42)
mx = np.empty(1000)
for b in range(1000):
    idx = rng.randint(0, n, n)
    a = _sh(A[idx])
    mx[b] = max(_sh(M[i][idx]) - a - obs[i] for i in range(len(pts)))
for i in np.argsort(-obs)[:3]:
    print(f"max-тест: {pts[i]} Δ={obs[i]:.4f} p_adj={np.mean(mx >= obs[i]):.3f} (точек {len(pts)})")

R = "/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/"
a = pd.read_parquet(R + "sigma_pred_h1_FINAL.parquet")
b = pd.read_parquet(R + "sigma_pred_h1_final3.parquet")
m = a.merge(b, on=["date", "ticker"], suffixes=("_F", "_3"))
m = m[m.date.dt.year.isin([2023, 2024])]
dd = (m.sigma_pred_h1_F.astype(float) - m.sigma_pred_h1_3.astype(float)).abs()
print(f"2023–2024: max|Δσ| = {dd.max():.2e}, медиана {dd.median():.2e}, отн. max {(dd / m.sigma_pred_h1_3).max():.2e}")
