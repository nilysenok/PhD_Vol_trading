#!/usr/bin/env bash
# Обратная выгрузка (Ж-133 п. 6), запускается НА МАКЕ из каталога perenos/:
#   ./zabrat.sh user@host:perenos <каталог на Маке>
# Забирает ТОЛЬКО ledger/, tables/, logs/ (белый список, три отдельных rsync) — ни кода, ни данных, ни .venv на Мак.
# Ключ и адрес — ваши (ssh-конфиг), в репозиторий не попадают. PERENOS_RSYNC — другой rsync (brew), по умолчанию rsync из PATH.
set -euo pipefail
say() { printf '%s  %s\n' "$(date '+%F %T')" "$*"; }
fail() { printf 'FAIL: %s\n' "$*"; exit 1; }

src=${1:-}; dst=${2:-}
[ -n "$src" ] && [ -n "$dst" ] || fail "использование: ./zabrat.sh user@host:perenos <каталог на Маке>"
case "$dst" in *:*) fail "назначение — локальный каталог на Маке, не сервер" ;; esac
RSYNC=${PERENOS_RSYNC:-rsync}
mkdir -p "$dst"
for d in ledger tables logs; do
  say "забираю $d/ -> $dst/$d/"
  "$RSYNC" -a --human-readable "${src%/}/$d/" "$dst/$d/" || fail "rsync $d/ — вывод выше"
done
plohie=$(find "$dst" \( -name '*.py' -o -name '*.tar.gz' -o -name '*.sh' -o -path '*/data/*' -o -path '*/kod/*' -o -name '*test_locked*' \) -print | head -3)
[ -z "$plohie" ] || fail "в забранном есть код/данные (белый список нарушен): $plohie"
echo "OK zabrat: $(find "$dst" -type f | wc -l | tr -d ' ') файлов в $dst (ledger/, tables/, logs/)"
