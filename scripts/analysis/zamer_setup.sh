#!/usr/bin/env bash
# Замер стратегий гл. 4: песочница data/strat_zamer/sbx (копии скриптов и входов; основная копия только читается).
set -euo pipefail
W="$(cd "$(dirname "$0")/../.." && pwd)"
MS="/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/moex_volatility"
S="$W/data/strat_zamer/sbx/moex_volatility"
mkdir -p "$S/scripts" "$S/results/final/strategies/data" "$S/results/final/strategies/walkforward_v4" \
         "$S/results/final/strategies/test_phases" "$W/data/strat_zamer/inputs"
for f in archivist strategies_walkforward daily_upper_bound stage3_walkforward stage3c_bcd_30phases \
         stage5_msp stage7_bootstrap stage8_economic fix_all_bootstrap_ce fix_msp_b; do
  cp "$MS/scripts/$f.py" "$S/scripts/"
done
D="$MS/results/final/strategies/data"
for f in ohlcv_daily_full ohlcv_hourly_full vpred_aligned; do
  cp "$D/$f.parquet" "$S/results/final/strategies/data/"
done
cp "$MS/results/final/strategies/walkforward_v4/narrowed_grid_v2.json" "$S/results/final/strategies/walkforward_v4/"
cp "$MS/results/final/strategies/test_phases/FINAL_best_configs.json" "$S/results/final/strategies/test_phases/"
md5 "$D"/vpred_aligned*.parquet "$S/results/final/strategies/data/vpred_aligned.parquet"
python3 "$W/scripts/analysis/zamer_keyrate.py" "$MS" "$W"
sed -i '' "s#Path(\"/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet\")#Path(\"$W/data/strat_zamer/inputs/cbr_key_rate.parquet\")#" "$S"/scripts/*.py
grep -n "cbr_key_rate" "$S"/scripts/*.py | cut -c1-220
