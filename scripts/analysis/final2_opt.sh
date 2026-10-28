#!/usr/bin/env bash
# Финальная сборка второго круга, шаг 1: optuna-30 (seed 42) на features_pit_v3 для h=5 —
# F2_h5 (V9 + Д8.12 + Д7 + Huber + mono −1 на rv/rv_d/rv_w/rv_m/bv), F2m_h5 (то же без Huber). Без nice (поручение дирижёра).
set -u
cd "$(dirname "$0")/../.."
R=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen
L=$R/final2/logs
mkdir -p "$L"
export L2_FEAT=$R/features_pit_v3.parquet
nohup python3 scripts/analysis/tune_L2_optuna.py --h 5 --cfg '{"v9": true, "harf": true, "huber": true, "mono": true, "mono_sign": -1}' \
  --blocks d7 --workers 5 --tag F2_h5 > "$L/opt_F2_h5.log" 2>&1 &
echo $! > "$L/opt_F2_h5.pid"
nohup python3 scripts/analysis/tune_L2_optuna.py --h 5 --cfg '{"v9": true, "harf": true, "mono": true, "mono_sign": -1}' \
  --blocks d7 --workers 5 --tag F2m_h5 > "$L/opt_F2m_h5.log" 2>&1 &
echo $! > "$L/opt_F2m_h5.pid"
