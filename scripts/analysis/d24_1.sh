#!/usr/bin/env bash
# Д24.1: очередь на 2 воркерах (абляция на точках финалистов -> переподбор B v2 с mono24). Логи — data/regen/D24_logs/.
set -u
cd "$(dirname "$0")/../.."
LG=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/D24_logs
mkdir -p "$LG"
P="python3 -W ignore scripts/analysis/d24_1.py"
$P fixed --cand D12_h1_A_d812 --tag D24_1_h1_d812mf --workers 2 > "$LG/fixed_h1.log" 2>&1
$P fixed --cand D16_5_h22 --tag D24_1_h22_165mf --workers 2 > "$LG/fixed_h22.log" 2>&1
$P fixed --cand D12_h5 --tag D24_1_h5_D12mf --workers 2 > "$LG/fixed_h5.log" 2>&1
$P tune --cand D12_h1_A_d812 --tag D24_1_h1_d812m --workers 2 > "$LG/tune_h1.log" 2>&1
$P tune --cand D16_5_h22 --tag D24_1_h22_165m --workers 2 > "$LG/tune_h22.log" 2>&1
$P tune --cand D12_h5 --tag D24_1_h5_D12m --workers 2 > "$LG/tune_h5.log" 2>&1
echo done > "$LG/d24_1.done"
