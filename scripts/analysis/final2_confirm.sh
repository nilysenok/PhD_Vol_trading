#!/usr/bin/env bash
# Финальная сборка второго круга: ЕДИНСТВЕННОЕ открытие подтверждающего набора (val 2022-2024, шаги 7-9, OOS) для финалистов dev_log № 48.
# --confirm-next 0: дальше по рейтингу не идти. Выходы — data/regen/final2/confirm/<кандидат>/confirm_*.csv; журнал — final2/gibrid_devlog_confirm_<кандидат>.md.
set -u
cd "$(dirname "$0")/../.."
R=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen
one() {
  local b=$1 h=$2 f=$3
  python3 -W ignore scripts/analysis/gibrid_v17.py --boost "$b" --h "$h" --features $R/features_pit_v3.parquet --legs single top5 seeds3 wrap2 \
    --out $R/final2/confirm/"$b" --devlog $R/final2/gibrid_devlog_confirm_"$b".md --confirm --confirm-next 0 --finalists "$f" \
    > $R/final2/logs/confirm_"$b".log 2>&1
  echo "confirm $b h=$h: код $? $(date +%T)"
}
one D12_h1_A_d812 1 "h1:top5|V0_XGB|B-sm|L0" &
one D12_h1_B_b0.25 1 "h1:top5|V0_XGB|B-sm|L0" &
one D12_h1_V0 1 "h1:top5|V0_XGB|B-sm|L0" &
one F2m_h5 5 "h5:wrap2|V0_LGB|B-smrob|L0" &
one D12_h5 5 "h5:wrap2|V0_XGB|B-sm|L0" &
one L2v3_h5 5 "h5:single|V0_XGB|B-smrob|L0" &
one D12_h22 22 "h22:seeds3|V0_XGB|B-smrob|L0,h22:single|minimax|B-smrob|L0.25|log,h22:single|step|B-smrob|L0.5" &
wait
