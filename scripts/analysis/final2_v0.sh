#!/usr/bin/env bash
# Финальная сборка второго круга: чистый V0 (бустинг на ln RV, без HAR-J) на features_pit_v3 для h=5 и h=22 при точках Bv2b
# (как контроль h=1 агента features-v3) — кандидат V0 с той же обёрткой и цель владельца (3). Без нового поиска.
set -u
cd "$(dirname "$0")/../.."
R=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen
for h in 5 22; do
  RERUN_H=$h python3 scripts/analysis/rerun_h1_v3.py --tag Bv3 --max-bin 255 --features $R/features_pit_v3.parquet --workers 3 \
    > $R/final2/logs/v0_Bv3_h$h.log 2>&1 &
done
wait
tail -1 $R/final2/logs/v0_Bv3_h5.log $R/final2/logs/v0_Bv3_h22.log
