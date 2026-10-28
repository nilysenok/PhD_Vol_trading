#!/usr/bin/env bash
# Д25.1: критерии c1–c4 через гибрид (gibrid_v17.py, --dev-eval oos, --etalon same, шаги 2-6; подтверждающий НЕ открывается, тест не строится).
# Аргументы: boost:h ...; не больше 2 процессов одновременно.
set -u
cd "$(dirname "$0")/../.."
R=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen
mkdir -p $R/D25/gibrid
one() {
  local b=$1 h=$2
  python3 -W ignore scripts/analysis/gibrid_v17.py --boost "$b" --h "$h" --features $R/features_pit_v3.parquet --dev-eval oos --etalon same \
    --legs ${LEGS:-single} --blend rv --regen $R/D25 --out $R/D25/gibrid/"$b" --devlog $R/D25/gibrid/devlog_"$b".md > $R/D25/gibrid/"$b".log 2>&1
  echo "gibrid $b h=$h: код $? $(date +%T)"
}
i=0
for x in "$@"; do
  one "${x%:*}" "${x#*:}" &
  i=$((i + 1))
  if (( i % 2 == 0 )); then wait; fi
done
wait
