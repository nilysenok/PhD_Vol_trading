#!/usr/bin/env bash
# Итоговая сборка финалистов третьего круга (final3.py). Вывод — в data/regen/final3/logs/run_*.log; в терминал — короткие строки.
#   bash scripts/analysis/final3.sh dry      # сухой прогон на кандидатах второго круга + HAR-X (data/regen/final3_dry/, подтверждающий НЕ открывается)
#   bash scripts/analysis/final3.sh dev      # пул + гибрид кандидатов третьего круга + правило + запись финалистов в dev_log.md (коммит, push) — ДО открытия
#   bash scripts/analysis/final3.sh confirm  # ЕДИНСТВЕННОЕ открытие подтверждающего (val 2022-2024) для финалистов; второй раз не запустится (замок)
#   bash scripts/analysis/final3.sh build    # тестовые прогнозы финала шагов 1-10 без метрик -> test_locked/final3/; σ_pred(h=1)
#   bash scripts/analysis/final3.sh report   # docs/otchety/FINAL_DESIGN_chernovik.md (перезапись)
#   bash scripts/analysis/final3.sh all      # dev -> confirm -> build -> report (запуск после 03:00 по команде дирижёра; решение владельца 01:35)
set -euo pipefail
cd "$(dirname "$0")/../.."
R=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen
W=${WORKERS:-6}
run() {  # фаза final3.py, лог в файл
  local tag=$1; shift
  local d=$R/final3; [[ " $* " == *" --dry "* ]] && d=$R/final3_dry
  mkdir -p "$d/logs"
  python3 -W ignore scripts/analysis/final3.py "$@" --workers "$W" > "$d/logs/run_$tag.log" 2>&1 || { echo "ОШИБКА фазы $tag — $d/logs/run_$tag.log"; exit 1; }
  echo "фаза $tag: OK $(date +%T)"
}
dev() {
  run pool pool; run gibrid gibrid; run select select --zapis
  git add docs/otchety/dev_log.md
  git commit -q -m "Третий круг: финалисты зафиксированы до открытия подтверждающего набора ($(date '+%d.%m %H:%M'))" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" \
    && echo "dev_log: коммит $(git rev-parse --short HEAD)"
  git push -q 2>/dev/null && echo "dev_log: push OK" || echo "dev_log: push не прошёл (коммит локальный)"
}
case "${1:-dry}" in
  dry) for f in pool gibrid select build report; do run "dry_$f" "$f" --dry --only-krug2; done ;;
  dev) dev ;;
  confirm) run confirm confirm --confirm ;;
  build) run build build ;;
  report) run report report ;;
  all) dev; run confirm confirm --confirm; run build build; run report report ;;
  *) echo "фаза: dry | dev | confirm | build | report | all"; exit 2 ;;
esac
