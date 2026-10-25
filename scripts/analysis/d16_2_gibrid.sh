#!/usr/bin/env bash
# Д16.2: критерии 1-4 мета-модели через гибрид (gibrid_v17.py, --dev-eval oos, шаги 2-6, одиночная модель-кандидат V0_XGB).
# Подтверждающий набор НЕ открывается (без --confirm), тест не строится (boost закрыт). Аргументы: пары <boost>:<h>; по два параллельно.
set -u
cd "$(dirname "$0")/../.."
R=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen
mkdir -p $R/D16_2/logs
one() {
  local b=${1%%:*} h=${1##*:}
  python3 -W ignore scripts/analysis/gibrid_v17.py --boost "$b" --h "$h" --features $R/features_pit_v3.parquet \
    --legs single --out $R/D16_2/gibrid/"$b" --devlog $R/D16_2/gibrid_devlog_"$b".md > $R/D16_2/logs/gibrid_"$b".log 2>&1
  echo "gibrid $b h=$h: код $? $(date +%T)"
}
N=0
for x in "$@"; do
  one "$x" &
  N=$((N + 1))
  if [ $((N % 2)) -eq 0 ]; then wait; fi
done
wait
