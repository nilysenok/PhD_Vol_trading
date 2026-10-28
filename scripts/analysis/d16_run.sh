#!/usr/bin/env bash
# Д16: HAR-X (шаги 1-10) -> Д16.1 (V9/Д8.12 на HAR-X, шаги 1-6; порог по 3 seed; выживший h=1 - шаги 7-10 для теста) -> оценка.
set -u
cd "$(dirname "$0")/../.."
L=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/D16
mkdir -p "$L"
if [ "${1:-}" != "--skip-harx" ]; then
  python3 scripts/analysis/harx.py --leak-test > "$L/leak.log" 2>&1
  python3 scripts/analysis/harx.py > "$L/harx.log" 2>&1
  python3 scripts/analysis/harx.py --no-d17 --tag HARX0 > "$L/harx0.log" 2>&1
fi
python3 scripts/analysis/d16_v9.py --workers 3 > "$L/d16_1.log" 2>&1 &
P1=$!
python3 scripts/analysis/d16_v9.py --workers 3 --cands h1_d812x h5_D12x h5_F2mx --seeds 43 44 --suffix _s4344 > "$L/d16_1_s4344.log" 2>&1
wait $P1
python3 scripts/analysis/d16_v9.py --workers 3 --cands h1_d812x --steps 7 8 9 --suffix _s79 > "$L/d16_1_s79.log" 2>&1 &
P2=$!
python3 scripts/analysis/d16_v9.py --workers 1 --cands h1_d812x --steps 10 --suffix _s10 > "$L/d16_1_s10.log" 2>&1
wait $P2
python3 scripts/analysis/d16_eval.py > "$L/eval.log" 2>&1
python3 scripts/analysis/d16_eval.py --seeds > "$L/seeds.log" 2>&1
echo done > "$L/run.done"
