#!/usr/bin/env bash
# Д16.1 h=1 (кандидат третьего круга D16_h1_d812x): слот LGBM (top5, seeds 43/44), XGB seeds 43/44 шагов 7-9, шаг 10 обеих моделей (seed 42/43/44).
set -u
cd "$(dirname "$0")/../.."
L=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/D16
V="python3 scripts/analysis/d16_v9.py --cands h1_d812x"
$V --models lgbm --steps 1 2 3 4 5 6 7 8 9 --workers 3 --suffix _L > "$L/h1_L.log" 2>&1 &
P=$!
$V --models lgbm --steps 1 2 3 4 5 6 7 8 9 --seeds 43 44 --workers 3 --suffix _Ls4344 > "$L/h1_Ls4344.log" 2>&1
wait $P
$V --models xgb --steps 7 8 9 --seeds 43 44 --workers 3 --suffix _s4344b > "$L/h1_s4344b.log" 2>&1 &
P=$!
$V --models xgb --steps 10 --seeds 43 44 --workers 2 --suffix _s10b > "$L/h1_s10b.log" 2>&1
$V --models lgbm --steps 10 --seeds 42 43 44 --workers 3 --suffix _Ls10 > "$L/h1_Ls10.log" 2>&1
wait $P
echo done > "$L/h1_full.done"
