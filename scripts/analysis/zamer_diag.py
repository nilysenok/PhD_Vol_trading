import sys
from pathlib import Path
import pandas as pd

sbx = Path(sys.argv[1])
old_dir = Path("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/moex_volatility/results/final/strategies/msp/perstrat_daily_returns")
for a in "ACD":
    n = pd.read_csv(sbx / f"results/final/strategies/msp/perstrat_daily_returns/MSP_{a}_inst.csv", index_col=0, parse_dates=True)
    o = pd.read_csv(old_dir / f"MSP_{a}_inst.csv", index_col=0, parse_dates=True)
    d = (n - o)
    nz = d[(d.abs() > 1e-9).any(axis=1)]
    print(a, "дней с расхождением:", len(nz), "по годам:", nz.groupby(nz.index.year).size().to_dict())
    print(nz[["trading_ret", "exposure"]].head(4).to_string())
