#!/bin/zsh
# Единственный тестовый прогон главы 4 (2022–2026). НЕ ЗАПУСКАТЬ до «готово к тесту» владельца и
# docs/otchety/FINAL_DESIGN_strategies.md (замок fastbt.dev.check_scope; переименование — дирижёр).
# σ_pred h=1: data/regen/sigma_pred_h1_final3_full.parquet (код главы 3, ключ vpred в fastbt/designs/final/*.yaml);
# HAR-J h=1: data/regen/sigma_pred_h1_harj_final2.parquet (= √har_corr теста главы 3 побитно).
# Решения владельца 10:00/10:05: метрики 2022–2026 (+ 2025–01.2026); окна калибровки заморожены на 2018–2021
# (--val-start 2018, val_end 2021); основной p — Ледуа–Вольф, блок 5, 10 000 повторов, IID — справочно;
# C — одна точка C1 W504 p70 h=5, остальная сетка C — чувствительность; A — 2022–2026.
# Никаких git-операций. Таблицы → docs/tables/strat_test/, лог → docs/otchety/ch4_test_run.log.
# Остановка на первой ошибке (set -e, pipefail).
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
M=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading
HARJ=$M/data/regen/sigma_pred_h1_harj_final2.parquet
cd $R
[[ -f docs/otchety/FINAL_DESIGN_strategies.md ]] || { echo "замок: нет FINAL_DESIGN_strategies.md"; exit 1; }
[[ -f $HARJ ]] || { echo "нет σ_pred HAR-J: $HARJ"; exit 1; }
export NUMBA_CACHE_DIR=$M/data/fastbt/numba_cache
export CH4_TABLES=$R/docs/tables/strat_test          # final_test, chapter4_a, decomp
export S9_TABLES=$R/docs/tables/strat_test           # fastbt.s9 --scope test
mkdir -p $CH4_TABLES
L=$R/docs/otchety/ch4_test_run.log
exec > >(tee $L) 2>&1
echo "ch4_test_run: $(date)"
python3 -m fastbt.final_test --main fastbt/designs/final/main.yaml \
  --extra d_binary=fastbt/designs/final/d_binary.yaml d_cont=fastbt/designs/final/d_cont.yaml \
  --harj-vpred $HARJ --val-start 2018 --boot-reps 10000 --workers 10
python3 -m fastbt.s9 sources --scope test --val-start 2018 --boot-reps 10000
python3 -m fastbt.s9 cache --scope test --val-start 2018 --boot-reps 10000 --workers 3
python3 -m fastbt.s9 run --scope test --val-start 2018 --boot-reps 10000
python3 -m fastbt.chapter4_a --workers 10 --boot-reps 10000
python3 -m fastbt.chapter4_a --workers 10 --boot-reps 10000 --leak-mask-old
python3 -m fastbt.decomp
echo "ch4_test_run: готово $(date)"
