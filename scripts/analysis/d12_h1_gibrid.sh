#!/bin/sh
# Д12 h=1: правило гибрида (gibrid_v17, критерии 1-4, эталон — модели с той же s²) по каждому кандидату; HAR-J — на v3.
# sh scripts/analysis/d12_h1_gibrid.sh A_d812 B_b0.25 ...
R=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen
cd "$(dirname "$0")/../.."
for c in "$@"; do
  F=$R/features_pit_v3.parquet
  python3 -W ignore scripts/analysis/gibrid_v17.py --boost D12_h1_$c --h 1 --features $F --legs single top5 seeds3 \
    --out $R/D12_h1_gibrid/$c --devlog $R/D12_h1_gibrid/gibrid_devlog.md > $R/D12_h1_logs/g_$c.log 2>&1
  echo "$c done" >> $R/D12_h1_logs/g_done
done
