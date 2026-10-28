#!/usr/bin/env bash
# Д22: критерии c1–c4 через гибрид (gibrid_v17.py, --dev-eval oos, --etalon same, шаги 2-6; подтверждающий НЕ открывается, тест не строится).
# h=1 — D22_h1 (ноги single/top5/seeds3); h=5/22 — D22_h{5,22} (одиночная модель-кандидат в обоих слотах; ноги single/seeds3).
set -u
cd "$(dirname "$0")/../.."
R=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen
mkdir -p $R/D22/gibrid
one() {
  local b=$1 h=$2; shift 2
  nice -n 5 python3 -W ignore scripts/analysis/gibrid_v17.py --boost "$b" --h "$h" --features $R/features_pit_v3.parquet --dev-eval oos --etalon same \
    --legs "$@" --out $R/D22/gibrid/"$b" --devlog $R/D22/gibrid/devlog_"$b".md > $R/D22/gibrid/"$b".log 2>&1
  echo "gibrid $b h=$h: код $? $(date +%T)"
}
one D22_h1 1 single top5 seeds3 &
one D22_h5 5 single seeds3 &
one D22_h22 22 single seeds3 &
wait
