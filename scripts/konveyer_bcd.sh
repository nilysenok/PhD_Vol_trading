#!/usr/bin/env bash
# Конвейер B/C/D главы 4 (T3): к2, к3, лестница σ, Δ к A, p ЛВ — одна команда перезапуска.
#   scripts/konveyer_bcd.sh <A-yaml> [--out DIR] [--reps N] [--n-shift N] [--c-grid YAML] [--pometka "..."]
# Текущий A — fastbt/designs/final/main.yaml; после фиксации A′ — его yaml (и --pometka без слова «текущий»).
# Только train 2018–2019, без выбора точек. Процессов не больше 2 (--workers 2 по умолчанию, потоки numba = 2).
set -euo pipefail
if [ $# -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  sed -n '2,5p' "$0"
  echo "использование: scripts/konveyer_bcd.sh <A-yaml> [опции]"
  exit 0
fi
cd "$(dirname "$0")/.."
uptime
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-2}"
exec python3 -m fastbt.konveyer_bcd "$@"
