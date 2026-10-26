#!/bin/zsh
# Полный прогон справочных LSTM/GRU (ref_nn_sv.py): сначала разработочный набор (исследования 1-5 -> переобучения 1-6), затем
# закрытые шаги (исследования 6-9 -> переобучения 7-10). Логи — в общий data/run/. Воркеров по умолчанию 5 (лимит 6 процессов).
W=${1:-5}
D=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/run
S=$(dirname $0)/ref_nn_sv.py
mkdir -p $D
python3 -W ignore $S --phase tune --steps 1 2 3 4 5 --workers $W >> $D/ref_nn_tune_dev.log 2>&1 &&
python3 -W ignore $S --phase refit --steps 1 2 3 4 5 6 --workers $W >> $D/ref_nn_refit_dev.log 2>&1 &&
python3 -W ignore $S --phase assemble >> $D/ref_nn_assemble.log 2>&1 &&
python3 -W ignore $S --phase tune --steps 6 7 8 9 --workers $W >> $D/ref_nn_tune_lock.log 2>&1 &&
python3 -W ignore $S --phase refit --steps 7 8 9 10 --workers $W >> $D/ref_nn_refit_lock.log 2>&1 &&
python3 -W ignore $S --phase assemble >> $D/ref_nn_assemble.log 2>&1
echo "$(date '+%F %T') ref_nn_sv.sh код $?" >> $D/ref_nn_assemble.log
