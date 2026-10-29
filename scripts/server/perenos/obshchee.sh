# Общее для скриптов каталога perenos/ (Ж-133): подключается через `. obshchee.sh`, сам не запускается.
# Все пути — от каталога скрипта; ничего вне него не читается и не пишется (кроме системного Python, из которого строится .venv).
set -euo pipefail
P=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$P"
export LC_ALL=C

say() { printf '%s  %s\n' "$(date '+%F %T')" "$*"; }
fail() { printf 'FAIL: %s\n' "$*"; exit 1; }

sha_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

sha_check() {   # $1 — каталог, $2 — файл сумм (путь от каталога); молча, код возврата
  if command -v sha256sum >/dev/null 2>&1; then (cd "$1" && sha256sum -c --quiet "$2"); else (cd "$1" && shasum -a 256 -c --quiet "$2"); fi
}

sha_spisok() {   # $1 — каталог; суммы всех файлов, пути ./… по порядку (формат sha256sum)
  (cd "$1" && find . -type f -print | sort | while IFS= read -r f; do printf '%s  %s\n' "$(sha_file "$f")" "$f"; done)
}

kv() {   # значение из VERSIYA (не «KOD»: на macOS имена без учёта регистра, рядом каталог kod/); stderr — чтобы было видно
  [ -f "$P/VERSIYA" ] || { echo "FAIL: нет файла VERSIYA — каталог perenos/ не собран или скопирован не целиком" >&2; exit 1; }
  sed -n "s/^$1=//p" "$P/VERSIYA"
}

KOD_SHA=$(kv tar_sha256)
KOD_SHORT=$(kv commit_short)
KOD_TAR="$P/kod_$KOD_SHORT.tar.gz"
KOD_DIR="$P/kod/$KOD_SHA"
KOD_SUMS="$P/kod/$KOD_SHA.SHA256SUMS"
ETALONY="$P/etalony_$KOD_SHORT.npz"
MANIFEST="$P/manifest_dannyh.tsv"
PY="$P/.venv/bin/python3"

yadra() { nproc 2>/dev/null || sysctl -n hw.ncpu; }

processov() {   # ядра − 4 (Ж-131 п. 4), не меньше 1; PERENOS_PROCS — явное число (локальная проверка)
  local n=${PERENOS_PROCS:-$(( $(yadra) - 4 ))}
  [ "$n" -ge 1 ] 2>/dev/null || n=1
  echo "$n"
}

okruzhenie() {   # данные — только perenos/data, кэши — perenos/cache; переменные, ведущие наружу, сняты
  unset FASTBT_DATA_ROOT FASTBT_VPRED FASTBT_Z2_VPRED FASTBT_Z2_R2FINAL FASTBT_NO_CACHE PYTHONHOME || true
  export PYTHONPATH="$KOD_DIR" FASTBT_MAIN="$P/data" FASTBT_BARS1M="$P/data/data/regen/D25/bars1m"
  export FASTBT_CACHE="$P/cache/fastbt" FASTBT_E1_CACHE="$P/cache/etap1" FASTBT_FS_CACHE="$P/cache/fs"
  export FASTBT_BARS5M_CACHE="$P/cache/bars5m" NUMBA_CACHE_DIR="$P/cache/numba/$KOD_SHA" PYTHONPYCACHEPREFIX="$P/cache/pyc"
  [ "${PERENOS_SYSTEM_SITE:-0}" = 1 ] || export PYTHONNOUSERSITE=1
  mkdir -p "$P/cache" "$P/logs" "$P/ledger" "$P/tables" "$P/proverka"
}

kod_cel() {   # tar тот же, распакованный код не правился (Ж-131 п. 6)
  [ -f "$KOD_TAR" ] || fail "нет $KOD_TAR"
  [ "$(sha_file "$KOD_TAR")" = "$KOD_SHA" ] || fail "sha256 $KOD_TAR ≠ VERSIYA (tar_sha256) — код подменён или скопирован не целиком"
  [ -d "$KOD_DIR" ] && [ -f "$KOD_SUMS" ] || fail "код не распакован — сначала ./razvernut.sh"
  sha_check "$KOD_DIR" "$KOD_SUMS" >/dev/null 2>&1 || fail "код в kod/$KOD_SHA изменён после распаковки — правка кода на сервере запрещена (Ж-131 п. 6); заново: ./razvernut.sh после удаления kod/"
  [ -z "$(find "$KOD_DIR" -perm -u+w -type f -print -quit)" ] || fail "в kod/$KOD_SHA есть файлы с правом записи — каталог кода только для чтения"
}
