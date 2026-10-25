#!/bin/bash
# М7 (лист решений 25.10): B на features_pit, h=1, в очереди после текущих тяжёлых задач.
# Ждёт: нет чужих tune_B.py / regen_boosting.py (основной прогон B, --extra, --watch), готовы базовые B_*_h1_val.parquet,
# 1-мин нагрузка < LOAD_MAX. Потом h=1; если |сдвиг val QLIKE (шаги 1–6)| >= 2 % у любой модели — h=5 и h=22.
# Запуск из корня копии: nohup bash scripts/analysis/m0_B_pit_queue.sh > data/run/m0_B_pit_queue.log 2>&1 &
cd "$(dirname "$0")/../.." || exit 1
BASE=${BASE:-/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/.claude/worktrees/agent-afcfa1385cb335411/data/regen}
W=${W:-4}
LOAD_MAX=${LOAD_MAX:-10}
FEAT=data/regen/features_pit.parquet
mkdir -p data/run
echo $$ > data/run/m0_B_pit_queue.pid
busy() { pgrep -lf "tune_B.py|regen_boosting.py" | grep -v -- "--tag" | grep -q . ; }
load1() { sysctl -n vm.loadavg | awk '{print int($2)}'; }
echo "$(date '+%F %T') ожидание очереди"
while busy || [ ! -f "$BASE/B_xgb_h1_val.parquet" ] || [ ! -f "$BASE/B_lgbm_h1_val.parquet" ] || [ "$(load1)" -ge "$LOAD_MAX" ]; do
  sleep 300
done
run() {  # $1 = список h
  echo "$(date '+%F %T') старт tune_B --hs $1 (workers $W)"
  OMP_NUM_THREADS=1 nice -n 10 python3 moex_volatility/scripts/tune_B.py --features "$FEAT" --tag Bpit --hs "$1" --workers "$W" || exit 2
}
run 1
python3 scripts/analysis/m0_B_pit_cmp.py --base "$BASE" --h 1
rc=$?
if [ $rc -eq 10 ]; then
  run 5,22
  python3 scripts/analysis/m0_B_pit_cmp.py --base "$BASE" --h 5
  python3 scripts/analysis/m0_B_pit_cmp.py --base "$BASE" --h 22
elif [ $rc -ne 0 ]; then
  echo "сравнение не выполнено (код $rc)"; exit 3
fi
echo "$(date '+%F %T') готово"
