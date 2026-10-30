#!/usr/bin/env bash
# Обратная выгрузка (Ж-133 п. 6), запускается НА МАКЕ из каталога perenos/:
#   ./zabrat.sh user@host:perenos <каталог на Маке>
# Забирает ТОЛЬКО (белый список): tables/, logs/; из ledger/ — manifest.json, checkpoint.json, SHA256SUMS*, daty.npy каждой
# стратегии и шарды p*/ только ledger 2.1 (manifest.json: "v": "2.1"; суммы, малы). Шарды ledger 2.0 (ряды, ~80 КБ на точку) остаются
# на сервере (↺ сужение Ж-131 п. 3, решение дирижёра volna_17 п. 3; откат — забрать 2.0 по запросу отдельной командой).
# Ни кода, ни данных, ни .venv на Мак. PERENOS_DRYRUN=1 — rsync -n (ничего не копирует, печатает что взял бы).
# Ключ и адрес — ваши (ssh-конфиг), в репозиторий не попадают. PERENOS_RSYNC — другой rsync (brew), по умолчанию rsync из PATH.
set -euo pipefail
say() { printf '%s  %s\n' "$(date '+%F %T')" "$*"; }
fail() { printf 'FAIL: %s\n' "$*"; exit 1; }

src=${1:-}; dst=${2:-}
[ -n "$src" ] && [ -n "$dst" ] || fail "использование: ./zabrat.sh user@host:perenos <каталог на Маке>"
case "$dst" in *:*) fail "назначение — локальный каталог на Маке, не сервер" ;; esac
RSYNC=(${PERENOS_RSYNC:-rsync} -a --human-readable)
[ "${PERENOS_DRYRUN:-0}" = 1 ] && RSYNC+=(-n -v)
src=${src%/}
mkdir -p "$dst"
for d in tables logs; do
  say "забираю $d/ -> $dst/$d/"
  "${RSYNC[@]}" "$src/$d/" "$dst/$d/" || fail "rsync $d/ — вывод выше"
done
say "забираю ledger/: метки (manifest.json, checkpoint.json, SHA256SUMS*, daty.npy)"
"${RSYNC[@]}" --prune-empty-dirs --include='*/' --include='manifest.json' --include='checkpoint.json' --include='SHA256SUMS*' \
  --include='daty.npy' --include='zamorozka.yaml' --include='morris_y.jsonl' --include='morris_*.csv' --exclude='*' "$src/ledger/" "$dst/ledger/" || fail "rsync ledger/ (метки) — вывод выше"
n21=0; n20=0
if [ "${PERENOS_DRYRUN:-0}" != 1 ]; then
  while IFS= read -r m; do
    sk=$(dirname "${m#"$dst"/ledger/}")
    if grep -Eq '"v": "2\.(1|2)"' "$m"; then   # явный список: 2.1 и 2.2 (2.2 = 2.1 + setka, rol, klyuch_n)
      say "ledger 2.1/2.2: $sk — шарды"
      "${RSYNC[@]}" --exclude='.zamok' "$src/ledger/$sk/" "$dst/ledger/$sk/" || fail "rsync ledger/$sk — вывод выше"
      n21=$((n21 + 1))
    else
      say "ledger 2.0 (или версия не 2.1): $sk — шарды остаются на сервере"
      n20=$((n20 + 1))
    fi
  done < <(find "$dst/ledger" -name manifest.json -print | sort)
fi
plohie=$(find "$dst" \( -name '*.py' -o -name '*.tar.gz' -o -name '*.sh' -o -path '*/data/*' -o -path '*/kod/*' -o -name '*test_locked*' \) -print | head -3)
[ -z "$plohie" ] || fail "в забранном есть код/данные (белый список нарушен): $plohie"
echo "OK zabrat: $(find "$dst" -type f | wc -l | tr -d ' ') файлов в $dst (tables/, logs/, ledger: 2.1 — $n21 стратегий с шардами, 2.0 — $n20 без шардов)"
