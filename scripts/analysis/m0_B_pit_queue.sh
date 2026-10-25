#!/bin/bash
# М7 (лист решений 25.10; дополнение 25.10): B на features_pit_v2, ТОЛЬКО h=1, в очереди после текущих тяжёлых задач.
# Ждёт: есть FEAT (v2); нет чужих tune_B.py / regen_boosting.py (основной прогон B, --extra, --watch);
# готовы базовые B_*_h1_val.parquet; 1-мин нагрузка < LOAD_MAX. Потом h=1 и сравнение (правило 2 %).
# h=5/22 — не запускаются: ждут решения владельца о цели (rv_target_h{h} или rv_target_mean_h{h}).
# Запуск из корня копии: nohup bash scripts/analysis/m0_B_pit_queue.sh > data/run/m0_B_pit_queue.log 2>&1 &
cd "$(dirname "$0")/../.." || exit 1
BASE=${BASE:-/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/.claude/worktrees/agent-afcfa1385cb335411/data/regen}
W=${W:-4}
LOAD_MAX=${LOAD_MAX:-10}
FEAT=${FEAT:-data/regen/features_pit_v2.parquet}
TAG=${TAG:-Bpit2}
mkdir -p data/run
echo $$ > data/run/m0_B_pit_queue.pid
busy() { pgrep -lf "tune_B.py|regen_boosting.py" | grep -v -- "--tag" | grep -q . ; }
load1() { sysctl -n vm.loadavg | awk '{print int($2)}'; }
RV1850=${RV1850:-/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/.claude/worktrees/agent-a51b95b5ae61fb6bc/data/iss_check/rv_1850_noquar.parquet}
if [ ! -f "$FEAT" ]; then  # v2 строится, как только агент ISS выложит rv_1850 (файл не меняется 2 мин)
  echo "$(date '+%F %T') жду $RV1850"
  until [ -f "$RV1850" ] && [ -z "$(find "$RV1850" -mmin -2)" ]; do sleep 120; done
  echo "$(date '+%F %T') сборка features_pit_v2"
  (cd scripts/analysis && nice python3 -W ignore m0_features_pit_v2.py --out "$OLDPWD/$FEAT") > data/run/m0_features_pit_v2.log 2>&1 || exit 4
fi
echo "$(date '+%F %T') ожидание очереди ($FEAT)"
while [ ! -f "$FEAT" ] || busy || [ ! -f "$BASE/B_xgb_h1_val.parquet" ] || [ ! -f "$BASE/B_lgbm_h1_val.parquet" ] || [ "$(load1)" -ge "$LOAD_MAX" ]; do
  sleep 300
done
echo "$(date '+%F %T') старт tune_B --hs 1 (workers $W)"
OMP_NUM_THREADS=1 nice -n 10 python3 moex_volatility/scripts/tune_B.py --features "$FEAT" --tag "$TAG" --hs 1 --workers "$W" || exit 2
python3 scripts/analysis/m0_B_pit_cmp.py --base "$BASE" --h 1 --tag "$TAG"
echo "код сравнения $? (10 = сдвиг >= 2 %: после решения о цели перегнать h=5/22; 0 = < 2 %: не перегонять)"
echo "$(date '+%F %T') готово"
