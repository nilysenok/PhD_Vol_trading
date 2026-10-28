#!/usr/bin/env bash
# Д25.2: критерии c1–c4 через гибрид (gibrid_v17.py, --dev-eval oos, --etalon same, шаги 2-6; подтверждающий НЕ открывается, тест не строится).
# Основные варианты — ноги single seeds3; seed-варианты (_s42/_s43/_s44) — один seed в single. Контроль «без рычага» — D15_h22 (тот же протокол).
# По 2 процесса одновременно (лимит Д25).
set -u
cd "$(dirname "$0")/../.."
R=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen
O=$R/D25
mkdir -p $O/gibrid
# seed-варианты контроля D15_h22 (производные копии по seed, исходные файлы не меняются)
python3 - <<'EOF'
import pandas as pd
R='/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen'
for m in ('xgb','lgbm'):
    s=pd.read_parquet(f'{R}/D15_h22_{m}_h22_val.parquet'); ss=pd.read_parquet(f'{R}/D15_h22_{m}_h22_seeds4344_val.parquet')
    c=['date','ticker','step','year','rv_actual','rv_pred']
    s[c].to_parquet(f'{R}/D25/D15_h22_s42_{m}_h22_val.parquet',index=False)
    for sd in (43,44):
        ss[ss.seed==sd][c].to_parquet(f'{R}/D25/D15_h22_s{sd}_{m}_h22_val.parquet',index=False)
EOF
one() {
  local b=$1 rg=$2; shift 2
  python3 -W ignore scripts/analysis/gibrid_v17.py --boost "$b" --h 22 --features $R/features_pit_v3.parquet --dev-eval oos --etalon same \
    --regen "$rg" --legs "$@" --out $O/gibrid/"$b" --devlog $O/gibrid/devlog_"$b".md > $O/gibrid/"$b".log 2>&1
  echo "gibrid $b: код $? $(date +%T)"
}
one D25_2A_h22 $O single seeds3 &
one D25_2B_h22 $O single seeds3 &
wait
[ -f $O/gibrid/D15_h22/designs_val_D15_h22_h22.csv ] || one D15_h22 $R single seeds3 &
one D25_2A_h22_s42 $O single &
wait
one D25_2A_h22_s43 $O single &
one D25_2A_h22_s44 $O single &
wait
one D25_2B_h22_s42 $O single &
one D25_2B_h22_s43 $O single &
wait
one D25_2B_h22_s44 $O single &
one D15_h22_s42 $O single &
wait
one D15_h22_s43 $O single &
one D15_h22_s44 $O single &
wait
