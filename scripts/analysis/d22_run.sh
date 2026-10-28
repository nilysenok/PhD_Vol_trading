#!/usr/bin/env bash
# Д22: полный прогон после подбора слоя 2 (d22_h1.py --tune): шаг 10, классификатор, hyb₁/r̂₁/m̂, абляции слоя 2, слои 3–4, оценка, абляции, детектор, экспорт.
# Логи — data/regen/D22/*.log. До 4 воркеров, nice 5.
set -u
cd "$(dirname "$0")/../.."
L=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/D22
Y="nice -n 5 python3 -W ignore"
if [ "${1:-}" != "--skip-l2" ]; then
  $Y scripts/analysis/d22_h1.py --step10 --workers 4 > $L/step10.log 2>&1
  $Y scripts/analysis/d22_h1.py --clf --workers 4 > $L/clf.log 2>&1
  $Y scripts/analysis/d22_h1.py --build > $L/build.log 2>&1
  $Y scripts/analysis/d22_h1.py --abl --workers 4 > $L/abl_l2.log 2>&1
fi
$Y scripts/analysis/d22_full.py --h1 > $L/eval_h1.log 2>&1
$Y scripts/analysis/d22_full.py --run > $L/run.log 2>&1
$Y scripts/analysis/d22_full.py --ablate > $L/ablate.log 2>&1
$Y scripts/analysis/d22_full.py --leak > $L/leak.log 2>&1
echo "done $(date +%T)" > $L/run.done
