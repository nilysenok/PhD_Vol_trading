#!/bin/bash
# Д15: база h (подбор или досчёт после обрыва), порог базы, кумулятивный проход вариантов. bash scripts/analysis/d15_chain.sh <h> [workers]
H=$1; W=${2:-3}
R=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen
D=$R/D15
S=$(cd "$(dirname "$0")" && pwd)
export L2_FEAT=$R/features_pit_v3.parquet
T=D15v_h${H}_base
if ! grep -q 'готов' "$D/base_h$H.log" 2>/dev/null; then
  if [ -f "$R/${T}_trials.csv" ]; then X=--resume; else X=; fi
  python3 "$S/d15_tune.py" --h "$H" --space B --cfg '{}' --tag "$T" --workers "$W" $X >> "$D/base_h$H.log" 2>&1 || exit 1
fi
if ! grep -q "$T,base" "$D/levers_h$H.csv" 2>/dev/null; then
  python3 "$S/d15_gate.py" --h "$H" --tag "$T" --label base
fi
python3 "$S/d15_driver.py" --h "$H" --workers "$W" ${3:+--only "$3"}
