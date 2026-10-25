#!/usr/bin/env bash
# Д16.7 / Д16.9: критерии c1-c4 — gibrid_v17 (oos, шаги 2-6, --etalon same), ноги single и seeds3; база — та же точка V9 на HAR-J с теми же ногами.
# Аргументы: пары <boost>:<h>. Одновременно — не больше GPAR (по умолчанию 2; перераспределение ядер 01:29).
set -u
cd "$(dirname "$0")"
R=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen
n=0
for x in "$@"; do
  b=${x%%:*}; h=${x##*:}
  python3 -W ignore gibrid_v17.py --boost "$b" --h "$h" --legs single seeds3 --features "$R/features_pit_v3.parquet" \
    --out "$R/D16_79/gibrid_$b" --devlog "$R/D16_79/gibrid_devlog_$b.md" > "$R/D16_79/logs/gibrid_$b.log" 2>&1 &
  n=$((n + 1))
  if [ $((n % ${GPAR:-2})) -eq 0 ]; then wait; fi
done
wait
