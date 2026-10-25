#!/usr/bin/env bash
# Финальная сборка второго круга, шаг 5: тестовые прогнозы (шаги 1-10) предварительных финалистов — БЕЗ МЕТРИК.
# Первый предварительный финалист h -> test_locked/final2/final_h{h}_test.parquet (заменить на финальный после открытия подтверждающего);
# остальные — test_locked/final2/cand/<h>_<порядок>_<кандидат>/. Схемы h=22 (minimax/step) — только шаги 1-9 (gibrid_v17 --build-test).
set -u
cd "$(dirname "$0")/../.."
R=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen
O=$R/test_locked/final2
B() { python3 -W ignore scripts/analysis/final2_build.py "$@" 2>&1 | grep -v -i warning; }
G() { python3 -W ignore scripts/analysis/gibrid_v17.py --features $R/features_pit_v3.parquet "$@" 2>&1 | grep -v -i warning; }
(
B --boost D12_h1_A_d812 --h 1 --design "top5|V0_XGB|B-sm|L0"
B --boost D12_h1_B_b0.25 --h 1 --design "top5|V0_XGB|B-sm|L0" --out $O/cand/h1_2_B_b0.25
B --boost D12_h1_V0 --h 1 --design "top5|V0_XGB|B-sm|L0" --out $O/cand/h1_3_V0
) &
(
B --boost F2m_h5 --h 5 --design "wrap2|V0_LGB|B-smrob|L0"
B --boost D12_h5 --h 5 --design "wrap2|V0_XGB|B-sm|L0" --out $O/cand/h5_2_D12_h5
B --boost L2v3_h5 --h 5 --design "single|V0_XGB|B-smrob|L0" --out $O/cand/h5_3_L2v3_h5
) &
(
B --boost D12_h22 --h 22 --design "seeds3|V0_XGB|B-smrob|L0"
B --boost D12_h22 --h 22 --design "single|V0_HAR|B-smrob|L0" --out $O/cand/h22_fallback_HARJ
G --boost D12_h22 --h 22 --legs single --build-test "single|minimax|B-smrob|L0.25|log" --test-out $O/cand/h22_2_minimax_steps1-9
G --boost D12_h22 --h 22 --legs single --build-test "single|step|B-smrob|L0.5" --test-out $O/cand/h22_3_step_steps1-9
) &
wait
echo "сборка тестовых прогнозов готова $(date +%T)"
