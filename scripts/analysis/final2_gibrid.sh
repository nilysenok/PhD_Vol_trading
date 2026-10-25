#!/usr/bin/env bash
# Финальная сборка второго круга, шаг 3: гибрид (gibrid_v17.py, --dev-eval oos, шаги 2-6) по каждому кандидату с ногами single/top5/seeds3/wrap2.
# Подтверждающий набор здесь НЕ открывается (без --confirm). Аргументы: пары <boost>:<h>; кандидаты считаются параллельно.
set -u
cd "$(dirname "$0")/../.."
R=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen
one() {
  local b=${1%%:*} h=${1##*:}
  python3 -W ignore scripts/analysis/gibrid_v17.py --boost "$b" --h "$h" --features $R/features_pit_v3.parquet \
    --legs single top5 seeds3 wrap2 --out $R/final2/gibrid/"$b" --devlog $R/final2/gibrid_devlog_"$b".md > $R/final2/logs/gibrid_"$b".log 2>&1
  echo "gibrid $b h=$h: код $? $(date +%T)"
}
for x in "$@"; do one "$x" & done
wait
