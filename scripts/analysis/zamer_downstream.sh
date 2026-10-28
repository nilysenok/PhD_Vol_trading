#!/usr/bin/env bash
# Нижняя часть пайплайна из сохранённых позиций A-D: stage5 -> ворота; затем fix_all_bootstrap_ce (бутстрап, CE).
set -euo pipefail
W="$(cd "$(dirname "$0")/../.." && pwd)"
S="$W/data/strat_zamer/sbx/moex_volatility"
MS="/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/moex_volatility/results/final/strategies"
mkdir -p "$S/results/final/strategies/walkforward" "$W/data/strat_zamer/logs"
for a in A B C D; do cp "$MS/walkforward/wf_positions_$a.parquet" "$S/results/final/strategies/walkforward/"; done
cd "$S/scripts"
s=$(date +%s); python3 stage5_msp.py > "$W/data/strat_zamer/logs/stage5.log" 2>&1; echo "stage5 $(( $(date +%s)-s )) c"
python3 "$W/scripts/analysis/zamer_gate_msp.py" "$S"
s=$(date +%s); python3 fix_all_bootstrap_ce.py > "$W/data/strat_zamer/logs/bootstrap_ce.log" 2>&1; echo "fix_all_bootstrap_ce $(( $(date +%s)-s )) c"
grep -E "C>A|D>A|B>A|D>C|γ|gamma" "$W/data/strat_zamer/logs/bootstrap_ce.log" | head -20
