#!/usr/bin/env python3
"""Шаг 0, стадия 3 (A/B/C/D): одна единица = тикер x стратегия x тестовый год (один уровень комиссии 0,0005).

Запуск: python3 zamer_unit_a.py <sbx/moex_volatility> <S1..S6> <TICKER> <год> [--prof]
Печатает время A, B, C, D для единицы; сверяет позиции A с сохранёнными wf_positions_A.parquet (основная копия).
Всё пишется только в песочницу (archivist/лог не вызываются: walkforward_* их не трогают, кроме log -> sbx).
"""
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sbx = Path(sys.argv[1])
sid, tk, yr = sys.argv[2], sys.argv[3], int(sys.argv[4])
prof = "--prof" in sys.argv
sys.path.insert(0, str(sbx / "scripts"))
import stage3_walkforward as m  # noqa: E402

(sbx / "results/final/strategies").mkdir(parents=True, exist_ok=True)
if '--fast' in sys.argv:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import zamer_fast
    zamer_fast.patch(m)
m.STRATEGIES = [sid]
m.TICKERS = [tk]
m.TEST_YEARS = [yr]

t0 = time.time()
daily_df, _, _ = m.load_data()
kr = pd.read_parquet(m.KR_PATH)
kr["date"] = pd.to_datetime(kr["date"])
kr_s = kr.set_index("date")["key_rate"].sort_index()
vp = pd.read_parquet(m.VPRED_PATH)
vp["date"] = pd.to_datetime(vp["date"])
td = m.prepare_ticker_data(tk, daily_df, kr_s, vp)
tdd = {tk: td}
t_prep = time.time() - t0
# numba warmup
m.warmup_numba()
_p = np.zeros(50); _p[10:15] = 1.0
m._apply_B_numba(_p, np.random.randn(50).cumsum() + 100, np.ones(50) * 0.02, 0.02, 1.0, 1.5, 15, 0.5, 1.0)
t_warm = time.time() - t0 - t_prep

comm = 0.0005
res = {}


def timed(name, fn, *a):
    pr = cProfile.Profile() if prof else None
    t = time.time()
    if pr:
        pr.enable()
    out = fn(*a)
    if pr:
        pr.disable()
    res[name] = time.time() - t
    if pr:
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(14)
        print(f"--- cProfile {name} ---\n" + "\n".join(s.getvalue().splitlines()[:40]))
    return out


grid = {}
fnA = m.walkforward_A_DISABLED if "--narrow" in sys.argv else m.walkforward_A
a_rows, a_pos, _ = timed("A", fnA, tdd, grid, comm)
b_rows, b_pos = timed("B", m.walkforward_B, tdd, a_pos, comm)
c_rows, c_pos = timed("C", m.walkforward_C, tdd, a_pos, comm)
d_rows, d_pos = timed("D", m.walkforward_D, tdd, a_pos, comm)

# сверка A с сохранёнными позициями
ref = pd.read_parquet("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/moex_volatility/results/final/strategies/walkforward/wf_positions_A.parquet")
ref = ref[(ref.strategy == sid) & (ref.ticker == tk) & (ref.year == yr)].set_index("date").position
cmp = {}
for nm, pd_ in (("A", a_pos), ("B", b_pos), ("C", c_pos), ("D", d_pos)):
    arr = pd_.get((sid, tk, yr))
    if arr is None:
        cmp[nm] = "нет"
        continue
    s = pd.Series(arr, index=td["dates"]).loc[ref.index] if len(ref) else None
    cmp[nm] = float(np.abs(s.values - ref.values).max()) if s is not None else None
print(f"UNIT{'(narrow)' if '--narrow' in sys.argv else ''} {sid} {tk} {yr}: prep={t_prep:.2f}s warm={t_warm:.2f}s "
      + " ".join(f"{k}={v:.2f}s" for k, v in res.items())
      + f" | A-row sharpe={a_rows[0]['sharpe'] if a_rows else None} skipped={a_rows[0].get('skipped') if a_rows else None}"
      + f" | maxabs(pos_A - saved A)={cmp['A']} (B,C,D vs saved A not comparable)")

tag = "fast" if "--fast" in sys.argv else "base"
outd = Path(__file__).resolve().parents[2] / "data/strat_zamer/logs"
np.savez(outd / f"ctrl_{tag}_{sid}_{tk}_{yr}.npz", A=a_pos.get((sid, tk, yr), np.zeros(1)), B=b_pos.get((sid, tk, yr), np.zeros(1)),
         C=c_pos.get((sid, tk, yr), np.zeros(1)), D=d_pos.get((sid, tk, yr), np.zeros(1)))
other = outd / f"ctrl_{'base' if tag == 'fast' else 'fast'}_{sid}_{tk}_{yr}.npz"
if other.exists():
    o = np.load(other)
    mine = np.load(outd / f"ctrl_{tag}_{sid}_{tk}_{yr}.npz")
    print("CTRL max|diff| A,B,C,D:", [float(np.nanmax(np.abs(o[k] - mine[k]))) if o[k].shape == mine[k].shape else "shape" for k in "ABCD"])
