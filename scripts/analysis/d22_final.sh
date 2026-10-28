#!/usr/bin/env bash
# Д22: финальный прогон слоёв 3–4 после решений владельца 02:15/02:25 (x9 = кандидаты Д24, λ += 100, без x5 на h=22):
# оценка, абляции (h=1 и h=5/22), сравнение ridge/QLIKE (Д24.12), детектор контура, экспорт файлов кандидатов, c1–c4 (gibrid_v17).
set -u
cd "$(dirname "$0")/../.."
L=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/D22
Y="nice -n 5 python3 -W ignore scripts/analysis/d22_full.py"
$Y --run > $L/run.log 2>&1
$Y --ablate > $L/ablate.log 2>&1
python3 -W ignore -c "import sys; sys.path.insert(0, 'scripts/analysis'); import d22_full as F; F.abl_h1(); F.cmp_obj()" > $L/cmp_obj.log 2>&1
$Y --leak > $L/leak.log 2>&1
$Y --export > $L/export.log 2>&1
bash scripts/analysis/d22_gibrid.sh > $L/gibrid_sh.log 2>&1
echo "done $(date +%T)" > $L/final.done
