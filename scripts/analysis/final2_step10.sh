#!/usr/bin/env bash
# Финальная сборка второго круга, шаг 5: шаг 10 (обучение <= 2024, val 2025, прогноз 01-02.2026) для всех кандидатов второго круга
# (финальный дизайн выбирается позже; модели дешёвые, поэтому считаются для всех). Метрики не считаются.
set -u
cd "$(dirname "$0")/../.."
R=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen
export L2_FEAT=$R/features_pit_v3.parquet
W=${W:-4}
run() { python3 scripts/analysis/final2_step10.py --workers $W "$@" 2>&1 | grep -v Warning; }
for c in A_d812 B_b0.25 V0; do run --kind h1 --boost D12_h1_$c --h 1; done
run --kind opt --boost D12_h22 --h 22 --cfg '{"v9": true, "linear_lambda": 1}'
run --kind opt --boost F2m_h5 --h 5 --cfg '{"v9": true, "harf": true, "mono": true, "mono_sign": -1}' --blocks d7
run --kind opt --boost F2_h5 --h 5 --cfg '{"v9": true, "harf": true, "huber": true, "mono": true, "mono_sign": -1}' --blocks d7
run --kind opt --boost D12_h5 --h 5 --cfg '{"v9": true, "harf": true, "huber": true}' --blocks d7
run --kind opt --boost L2v3_h5 --h 5 --cfg '{"v9": true, "harf": true}' --blocks d7
echo "шаг 10: все кандидаты готовы $(date +%T)"
