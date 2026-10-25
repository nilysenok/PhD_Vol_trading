#!/bin/zsh
# Единственный тестовый прогон главы 4 (2022–2026). НЕ ЗАПУСКАТЬ до «да» владельца и
# docs/otchety/FINAL_DESIGN_strategies.md (замок fastbt.dev.check_scope).
# σ_pred: гибрид Д8.12 final2 (2017–2026, ключ vpred в fastbt/designs/final/*.yaml); HAR-J (h=1)
# final2 — data/regen/sigma_pred_h1_harj_final2.parquet; скрипт рисунков из ветки figures-final.
set -e
R=$(cd "$(dirname "$0")/.." && pwd)
M=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading
HARJ=${HARJ:-$M/data/regen/sigma_pred_h1_harj_final2.parquet}
cd $R
[[ -f docs/otchety/FINAL_DESIGN_strategies.md ]] || { echo "замок: нет FINAL_DESIGN_strategies.md"; exit 1; }
[[ -f $HARJ ]] || { echo "нет σ_pred HAR-J: $HARJ"; exit 1; }
export NUMBA_CACHE_DIR=$M/data/fastbt/numba_cache
L=$M/data/run/ch4_test_$(date +%Y%m%d_%H%M%S).log
{
python3 -m fastbt.final_test --main fastbt/designs/final/main.yaml \
  --extra d_binary=fastbt/designs/final/d_binary.yaml d_cont=fastbt/designs/final/d_cont.yaml \
  --harj-vpred $HARJ --boot-reps 10000 --workers 10 --val-start 2017
python3 -m fastbt.chapter4_a --workers 10 --boot-reps 10000
python3 -m fastbt.chapter4_a --workers 10 --boot-reps 10000 --leak-mask-old
python3 -m fastbt.decomp
git checkout origin/figures-final -- scripts/analysis/
python3 scripts/analysis/figures_final.py \
  --series $M/data/fastbt/final_series/perstrat_daily_returns --out docs/tables/fig4_final
} 2>&1 | tee $L
