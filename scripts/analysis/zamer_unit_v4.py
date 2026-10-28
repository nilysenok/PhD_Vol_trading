#!/usr/bin/env python3
"""Замер и ворота движка v4 (strategies_walkforward_v4.py) на контрольном блоке: тикер x стратегия, только daily.

python3 zamer_unit_v4.py <sbx/moex_volatility> <S1..S6> <TICKER> [--prof] [--fast] [--years 2022,2023]
Исходный код v4 не правится. Время по подходам (A, B, C, D) считается вокруг функций approach_*_v4.
Ворота: доля совпавших позиций с daily_positions_pre2026.parquet (основная копия, чтение) по подходам и годам.
Маска как в коде ([1]: конец обучения = 1 янв Y+1) — режим по умолчанию.
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
sid, tk = sys.argv[2], sys.argv[3]
prof = "--prof" in sys.argv
sys.path.insert(0, str(sbx / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strategies_walkforward_v4 as v4  # noqa: E402

if "--fast" in sys.argv:
    import zamer_fast
    import strategies_walkforward as sw

v4.TIMEFRAMES = ["daily"]
v4.STRATEGY_IDS = [sid]
if "--years" in sys.argv:
    yrs = [int(x) for x in sys.argv[sys.argv.index("--years") + 1].split(",")]
    v4.A_TEST_YEARS = yrs
    v4.BCD_TEST_YEARS = yrs

T = {}
for nm in ("approach_a_v4", "approach_b_v4", "approach_c_v4", "approach_d_v4"):
    f = getattr(v4, nm)

    def wrap(f=f, nm=nm):
        def g(*a, **k):
            pr = cProfile.Profile() if prof else None
            t = time.time()
            if pr:
                pr.enable()
            r = f(*a, **k)
            if pr:
                pr.disable()
                s = io.StringIO()
                pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(10)
                print(f"--- cProfile {nm} (tottime) ---\n" + "\n".join(l[:160] for l in s.getvalue().splitlines()[:26]))
            T[nm[9]] = T.get(nm[9], 0) + time.time() - t
            return r
        return g
    setattr(v4, nm, wrap())

t0 = time.time()
daily, hourly, vpred = v4.load_data()
t_load = time.time() - t0
t0 = time.time()
out = v4.process_ticker_v4(tk, daily, hourly, vpred)
t_all = time.time() - t0
print(f"V4UNIT {sid} {tk} daily: load={t_load:.1f}s total={t_all:.1f}s " + " ".join(f"{k}={v:.1f}s" for k, v in sorted(T.items())))

pos = pd.DataFrame(out["positions"], columns=["date", "strategy", "tf", "ticker", "approach", "test_year", "position", "dgr"])
pos.to_parquet(Path(__file__).resolve().parents[2] / f"data/strat_zamer/logs/v4_{sid}_{tk}{'_fast' if '--fast' in sys.argv else ''}.parquet")
ref = pd.read_parquet("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/moex_volatility/results/final/strategies/walkforward_v4/daily_positions_pre2026.parquet")
ref = ref[(ref.tf == "daily") & (ref.ticker == tk) & (ref.strategy == v4.STRATEGY_NAMES[sid])]
m = pos.merge(ref, on=["date", "strategy", "tf", "ticker", "approach", "test_year"], suffixes=("", "_ref"), how="inner")
m["eq"] = (m.position - m.position_ref).abs() < 1e-6
g = m.groupby(["approach", "test_year"]).agg(n=("eq", "size"), match=("eq", "mean"), new_nz=("position", lambda s: (s != 0).mean()),
                                              ref_nz=("position_ref", lambda s: (s != 0).mean())).round(3)
print(g.to_string())
