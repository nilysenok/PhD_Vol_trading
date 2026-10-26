#!/usr/bin/env bash
# R2 Б2: очередь команд r2_roli_h5_h22.py по ролям и h (последовательно, до 5 воркеров в каждой).
# scripts/analysis/r2_B2_run.sh <команда> <лог> [роль:h ...]    пример: r2_B2_run.sh kontrol kontrol.log ost:5 gam:5
set -u
cd "$(dirname "$0")/../.."
CMD=$1; LOG=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/R2/cache/B2/$2; shift 2
: > "$LOG"
for rh in "$@"; do
  r=${rh%%:*}; h=${rh##*:}
  python3 -W ignore scripts/analysis/r2_roli_h5_h22.py "$CMD" --role "$r" --h "$h" --workers 5 ${EXTRA:-} >> "$LOG" 2>&1
  echo "$CMD $r $h exit $?" >> "$LOG"
done
echo "${CMD}_DONE $(date +%T)" >> "$LOG"
