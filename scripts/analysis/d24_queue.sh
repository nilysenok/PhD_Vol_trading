#!/usr/bin/env bash
# Д24a: общая очередь на 2 воркерах (после Д22/Д23). Порядок: абляции Д24.1 на точках -> Д24.3 (10 seed) -> Д24.2 -> переподбор Д24.1.
# Аргумент 1 (необязательно) — pid, завершения которого ждать перед стартом.
set -u
cd "$(dirname "$0")/../.."
LG=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/D24_logs
mkdir -p "$LG"
if [ "${1:-}" != "" ]; then while kill -0 "$1" 2>/dev/null; do sleep 5; done; fi
P1="python3 -W ignore scripts/analysis/d24_1.py"
P3="python3 -W ignore scripts/analysis/d24_3.py"
$P1 fixed --cand D16_5_h22 --tag D24_1_h22_165mf --workers 2 > "$LG/fixed_h22.log" 2>&1
$P1 fixed --cand D12_h5 --tag D24_1_h5_D12mf --workers 2 > "$LG/fixed_h5.log" 2>&1
for c in D16_5_h22 D15_h22 D17_1_h5_G5Xopt D12_h5 D16_h1_d812x D12_h1_A_d812 D12_h1_V0; do
  $P3 refit --cand $c --workers 2 > "$LG/seeds10_$c.log" 2>&1
done
$P3 table > "$LG/seeds10_table.log" 2>&1
python3 -W ignore scripts/analysis/d24_2.py run --workers 2 > "$LG/d24_2_run.log" 2>&1
python3 -W ignore scripts/analysis/d24_2.py gate > "$LG/d24_2_gate.log" 2>&1
$P1 tune --cand D12_h1_A_d812 --tag D24_1_h1_d812m --workers 2 > "$LG/tune_h1.log" 2>&1
$P1 tune --cand D16_5_h22 --tag D24_1_h22_165m --workers 2 > "$LG/tune_h22.log" 2>&1
$P1 tune --cand D12_h5 --tag D24_1_h5_D12m --workers 2 > "$LG/tune_h5.log" 2>&1
echo done > "$LG/queue.done"
